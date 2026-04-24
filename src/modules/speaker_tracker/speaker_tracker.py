"""
Speaker Tracking Module
Handles multi-speaker detection, tracking, and predictive switching for interviews/talks
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from tqdm import tqdm

from models import SpeakerSegment, TrackedSubject

# Minimum IoU threshold for matching detections across frames
IOU_THRESHOLD = 0.3
# Maximum distance (as fraction of frame) for matching
DISTANCE_THRESHOLD = 0.2
# Seconds before speaker change to start panning
PRE_SWITCH_TIME = 1.0
# Minimum frames to consider a subject "established"
MIN_TRACK_FRAMES = 5

class SpeakerTracker:
    """
    Tracks multiple speakers/subjects across video frames
    Correlates visual subjects with audio speaking patterns
    Implements predictive focus switching for interviews/talks
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize speaker tracker"""
        self.use_gpu = use_gpu
        self.tracked_subjects: Dict[int, TrackedSubject] = {}
        self.next_track_id = 0
        self.speaker_segments: List[SpeakerSegment] = []
        self.speaker_timeline: List[Dict] = [] # Timeline of who is speaking when
        
    def track_subjects(
        self,
        video_path: str,
        detection_results: Dict[str, Any],
        transcription: Optional[Dict] = None,
        sample_rate: int = 5
    ) -> Dict[str, Any]:
        """
        Track subjects throughout the video and correlate with speaking patterns
        
        Args:
            video_path: Path to video file
            detection_results: Results from SubjectDetector
            transcription: Transcription results with timestamps
            sample_rate: Frame sampling rate
            
        Return:
            Dictionary with tracking results
        """
        logger.info("Starting multi-subject tracking...")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Reset tracking state
        self.tracked_subjects = {}
        self.next_track_id = 0
        
        # Get detection timeline
        subject_timeline = detection_results.get('subject_timeline', [])
        detection_lookup = {entry['frame']: entry for entry in subject_timeline}
        
        # Analyze audio for speaker detection (if transcription available)
        speaker_patterns = self._analyze_speaker_patterns(transcription, fps, total_frames)
        
        frame_idx = 0
        prev_detections = []
        
        # Progress bar
        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing frames")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps
                
                # Get detections for this frame
                frame_detections = []
                if frame_idx in detection_lookup:
                    frame_detections = detection_lookup[frame_idx].get('subjects', [])
                    
                # Match detections to existing tracks
                self._update_tracks(frame_detections, timestamp, width, height)
                
                # Determine current speaker (if any)
                current_speaker = self._get_current_speaker(speaker_patterns, timestamp)
                
                # Update speaker timeline
                if current_speaker is not None:
                    self.speaker_timeline.append({
                        'timestamp': timestamp,
                        'frame': frame_idx,
                        'speaker_id': current_speaker,
                        'speaker_position': self._get_subject_position(current_speaker)
                    })
                    
                frame_idx += 1
                pbar.update(1)
                
        cap.release()

        # Finalize tracks
        self._finalize_tracks(fps)
        
        # Generate focus points with predictive switching
        focus_points = self._generate_focus_points(speaker_patterns, fps, total_frames, width, height)
        
        # Identify speakers (match subjects to speaking patterns)
        speaker_mapping = self._identify_speakers(speaker_patterns)
        
        result = {
            'tracked_subjects': {
                track_id: {
                    'track_id': ts.track_id,
                    'class_name': ts.class_name,
                    'first_seen': ts.first_seen,
                    'last_seen': ts.last_seen,
                    'total_frame': ts.total_frames,
                    'avg_position': ts.avg_position,
                    'screen_time': ts.screen_time,
                    'speaking_time': sum(s['end'] - s['start'] for s in ts.speaking_segments),
                    'positions': ts.positions[-50:] if ts.positions else [], # Last 50 positions
                }
                for track_id, ts in self.track_subjects.items()
            },
            'speaker_segments': [
                {
                    'speaker_id': ss.speaker_id,
                    'start': ss.start,
                    'end': ss.end,
                    'duration': ss.end - ss.start,
                    'transcript': ss.transcript,
                    'confidence': ss.confidence
                }
                for ss in self.speaker_segments
            ],
            'speaker_timeline': self.speaker_timeline,
            'speaker_patterns': speaker_patterns,
            'focus_points': focus_points,
            'speaker_mapping': speaker_mapping,
            'is_multi_speaker': len(self.track_subjects) >= 2,
            'subject_count': len(self.track_subjects),
        }
        
        logger.success(f"Tracking complete: {len(self.track_subjects)} subjects tracked")
        
        return result

    def _update_tracks(
        self,
        detections: List[Dict],
        timestamp: float,
        width: int,
        height: int
    ):
        """Update tracks with new detections"""
        matched_tracks = set()
        
        # Sort detections by score/importance
        detections = sorted(detections, key=lambda x: x.get('score', 0), reverse=True)
        
        for detection in detections:
            det_center = detection.get('center', (width // 2, height // 2))
            det_bbox = detection.get('bbox', (0, 0, 100, 100))
            det_class = detection.get('class', 'unknown')
            
            # Find best matching track
            best_match = None
            best_score = 0
            
            for track_id, track in self.tracked_subjects.items():
                if track_id in matched_tracks:
                    continue
                
                # Check if same class (or compatible classes)
                if not self._compatible_classes(track.class_name, det_class):
                    continue
                
                # Calculate match score based on position proximity
                if track.positions:
                    last_pos = track.positions[-1]['center']
                    distance = np.sqrt(
                        (det_center[0] - last_pos[0]) ** 2 + (det_center[1] - last_pos[1]) ** 2
                    )
                    max_dist = np.sqrt(width ** 2 + height ** 2) * DISTANCE_THRESHOLD
                    
                    if distance < max_dist:
                        score = 1 - (distance / max_dist)
                        if score > best_score:
                            best_score = score
                            best_match = track_id
                            
            if best_match is not None:
                