"""
Speaker Tracking Module
Handles multi-speaker detection, tracking, and predictive switching for interviews/talks
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass, field
from collections import defaultdict
import time

@dataclass
class TrackedSubject:
    """A tracked subject with persistent identity"""
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    positions: List[Dict] = field(default_factory=list)  # List of {timestamp, center, bbox}
    total_frames: int = 0
    avg_position: Tuple[float, float] = (0, 0)
    speaking_segments: List[Dict] = field(default_factory=list)  # Times when this subject is speaking
    screen_time: float = 0.0  # Total time on screen

@dataclass
class SpeakerSegment:
    """A segment where a specific speaker is talking"""
    speaker_id: int
    start: float
    end: float
    confidence: float
    transcript: str = ""

class SpeakerTracker:
    """
    Tracks multiple speakers/subjects across video frames
    Correlates visual subjects with audio speaking patterns
    Implements predictive focus switching for interviews/talks
    """
    
    # Minimum IoU threshold for matching detections across frames
    IOU_THRESHOLD = 0.3
    # Maximum distance (as fraction of frame) for matching
    DISTANCE_THRESHOLD = 0.2
    # Seconds before speaker change to start panning
    PRE_SWITCH_TIME = 1.0
    # Minimum frames to consider a subject "established"
    MIN_TRACK_FRAMES = 5
    
    def __init__(self, use_gpu: bool = True):
        """Initialize speaker tracker"""
        self.use_gpu = use_gpu
        self.tracked_subjects: Dict[int, TrackedSubject] = {}
        self.next_track_id = 0
        self.speaker_segments: List[SpeakerSegment] = []
        self.speaker_timeline: List[Dict] = []  # Timeline of who is speaking when
        
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
            
        Returns:
            Dictionary with tracking results
        """
        logger.info("Starting multi-subject tracking...")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
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
                    'total_frames': ts.total_frames,
                    'avg_position': ts.avg_position,
                    'screen_time': ts.screen_time,
                    'speaking_time': sum(s['end'] - s['start'] for s in ts.speaking_segments),
                    'positions': ts.positions[-50:] if ts.positions else [],  # Last 50 positions
                }
                for track_id, ts in self.tracked_subjects.items()
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
            'is_multi_speaker': len(self.tracked_subjects) >= 2,
            'subject_count': len(self.tracked_subjects),
        }
        
        logger.success(f"Tracking complete: {len(self.tracked_subjects)} subjects tracked")
        
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
                        (det_center[0] - last_pos[0]) ** 2 + 
                        (det_center[1] - last_pos[1]) ** 2
                    )
                    max_dist = np.sqrt(width ** 2 + height ** 2) * self.DISTANCE_THRESHOLD
                    
                    if distance < max_dist:
                        score = 1 - (distance / max_dist)
                        if score > best_score:
                            best_score = score
                            best_match = track_id
            
            if best_match is not None:
                # Update existing track
                track = self.tracked_subjects[best_match]
                track.last_seen = timestamp
                track.total_frames += 1
                track.positions.append({
                    'timestamp': timestamp,
                    'center': det_center,
                    'bbox': det_bbox
                })
                # Update average position
                self._update_avg_position(track)
                matched_tracks.add(best_match)
            else:
                # Create new track
                new_track = TrackedSubject(
                    track_id=self.next_track_id,
                    class_name=det_class,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    total_frames=1,
                    positions=[{
                        'timestamp': timestamp,
                        'center': det_center,
                        'bbox': det_bbox
                    }],
                    avg_position=det_center
                )
                self.tracked_subjects[self.next_track_id] = new_track
                self.next_track_id += 1
    
    def _compatible_classes(self, class1: str, class2: str) -> bool:
        """Check if two class names are compatible for tracking"""
        # Group similar classes
        person_classes = {'person', 'face', 'face_profile', 'body', 'upper_body'}
        
        if class1 == class2:
            return True
        if class1 in person_classes and class2 in person_classes:
            return True
        
        return False
    
    def _update_avg_position(self, track: TrackedSubject):
        """Update average position for a track"""
        if track.positions:
            avg_x = sum(p['center'][0] for p in track.positions) / len(track.positions)
            avg_y = sum(p['center'][1] for p in track.positions) / len(track.positions)
            track.avg_position = (avg_x, avg_y)
    
    def _analyze_speaker_patterns(
        self, 
        transcription: Optional[Dict], 
        fps: float,
        total_frames: int
    ) -> List[Dict]:
        """
        Analyze transcription to detect speaker patterns
        Uses pauses, segment patterns, and content to estimate speaker changes
        """
        patterns = []
        
        if not transcription:
            # No transcription, create default pattern
            duration = total_frames / fps
            patterns.append({
                'speaker_id': 0,
                'start': 0,
                'end': duration,
                'confidence': 0.5,
                'transcript': ''
            })
            return patterns
        
        segments = transcription.get('segments', [])
        
        if not segments:
            return patterns
        
        # Simple speaker detection based on pauses and segment patterns
        # In a real implementation, this would use actual diarization
        current_speaker = 0
        prev_end = 0
        speaker_segment_start = segments[0]['start'] if segments else 0
        current_transcript = ""
        
        for i, segment in enumerate(segments):
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            text = segment.get('text', '')
            
            # Check for speaker change indicators
            speaker_change = False
            
            # Long pause suggests speaker change
            pause = seg_start - prev_end
            if pause > 1.5:  # 1.5 second pause
                speaker_change = True
            
            # Very short segment might be interruption
            if seg_end - seg_start < 0.5 and i > 0:
                speaker_change = True
            
            # Save previous speaker segment if change detected
            if speaker_change and current_transcript:
                patterns.append({
                    'speaker_id': current_speaker,
                    'start': speaker_segment_start,
                    'end': prev_end,
                    'confidence': 0.7,
                    'transcript': current_transcript.strip()
                })
                
                # Switch speaker
                current_speaker = 1 - current_speaker  # Toggle between 0 and 1
                speaker_segment_start = seg_start
                current_transcript = ""
            
            current_transcript += " " + text
            prev_end = seg_end
        
        # Add final segment
        if current_transcript:
            patterns.append({
                'speaker_id': current_speaker,
                'start': speaker_segment_start,
                'end': prev_end,
                'confidence': 0.7,
                'transcript': current_transcript.strip()
            })
        
        # Store speaker segments
        self.speaker_segments = [
            SpeakerSegment(
                speaker_id=p['speaker_id'],
                start=p['start'],
                end=p['end'],
                confidence=p['confidence'],
                transcript=p['transcript']
            )
            for p in patterns
        ]
        
        logger.info(f"Detected {len(set(p['speaker_id'] for p in patterns))} speakers, {len(patterns)} segments")
        
        return patterns
    
    def _get_current_speaker(self, speaker_patterns: List[Dict], timestamp: float) -> Optional[int]:
        """Get the current speaker at a given timestamp"""
        for pattern in speaker_patterns:
            if pattern['start'] <= timestamp <= pattern['end']:
                return pattern['speaker_id']
        return None
    
    def _get_subject_position(self, subject_id: int) -> Optional[Tuple[float, float]]:
        """Get the average position of a subject"""
        if subject_id in self.tracked_subjects:
            return self.tracked_subjects[subject_id].avg_position
        return None
    
    def _finalize_tracks(self, fps: float):
        """Finalize track statistics"""
        for track_id, track in self.tracked_subjects.items():
            track.screen_time = track.last_seen - track.first_seen
            
            # Remove very short tracks (likely noise)
            if track.total_frames < self.MIN_TRACK_FRAMES:
                del self.tracked_subjects[track_id]
    
    def _generate_focus_points(
        self,
        speaker_patterns: List[Dict],
        fps: float,
        total_frames: int,
        width: int,
        height: int
    ) -> List[Dict]:
        """
        Generate focus points with predictive switching
        Switches focus 1 second before speaker change
        """
        focus_points = []
        
        if len(self.tracked_subjects) < 2:
            # Single subject or no subjects - use simple centering
            duration = total_frames / fps
            for t in np.arange(0, duration, 0.1):
                focus_points.append({
                    'timestamp': t,
                    'target_subject': 0,
                    'center': self._get_safe_center(0, width, height),
                    'is_transition': False,
                    'transition_progress': 0.0
                })
            return focus_points
        
        # Create speaker timeline with pre-switching
        speaker_changes = []
        for i in range(1, len(speaker_patterns)):
            prev_pattern = speaker_patterns[i - 1]
            curr_pattern = speaker_patterns[i]
            
            if prev_pattern['speaker_id'] != curr_pattern['speaker_id']:
                change_time = curr_pattern['start']
                speaker_changes.append({
                    'time': change_time,
                    'from_speaker': prev_pattern['speaker_id'],
                    'to_speaker': curr_pattern['speaker_id'],
                    'pre_switch_start': max(0, change_time - self.PRE_SWITCH_TIME)
                })
        
        # Generate focus points
        duration = total_frames / fps
        current_speaker = speaker_patterns[0]['speaker_id'] if speaker_patterns else 0
        transition_in_progress = False
        transition_start_time = 0
        transition_end_time = 0
        transition_from = 0
        transition_to = 0
        
        for t in np.arange(0, duration, 0.1):  # Every 100ms
            # Check for upcoming speaker change
            for change in speaker_changes:
                if t >= change['pre_switch_start'] and t < change['time']:
                    transition_in_progress = True
                    transition_start_time = change['pre_switch_start']
                    transition_end_time = change['time']
                    transition_from = change['from_speaker']
                    transition_to = change['to_speaker']
                    break
                elif t >= change['time'] and transition_in_progress:
                    transition_in_progress = False
                    current_speaker = change['to_speaker']
            
            # Calculate center position
            if transition_in_progress:
                # Smooth transition between speakers
                progress = (t - transition_start_time) / (transition_end_time - transition_start_time)
                progress = self._ease_in_out(progress)
                
                from_pos = self._get_subject_center(transition_from, t, width, height)
                to_pos = self._get_subject_center(transition_to, t, width, height)
                
                center_x = from_pos[0] * (1 - progress) + to_pos[0] * progress
                center_y = from_pos[1] * (1 - progress) + to_pos[1] * progress
                center = (int(center_x), int(center_y))
                
                focus_points.append({
                    'timestamp': t,
                    'target_subject': transition_to,
                    'center': center,
                    'is_transition': True,
                    'transition_progress': progress,
                    'from_subject': transition_from,
                    'to_subject': transition_to
                })
            else:
                # Focus on current speaker
                center = self._get_subject_center(current_speaker, t, width, height)
                focus_points.append({
                    'timestamp': t,
                    'target_subject': current_speaker,
                    'center': center,
                    'is_transition': False,
                    'transition_progress': 0.0
                })
        
        return focus_points
    
    def _get_subject_center(
        self, 
        speaker_id: int, 
        timestamp: float, 
        width: int, 
        height: int
    ) -> Tuple[int, int]:
        """Get the center position of a subject at a given timestamp"""
        # Find the subject that corresponds to this speaker
        for track_id, track in self.tracked_subjects.items():
            # For simplicity, assume track_id == speaker_id mapping
            # In real implementation, this would use speaker identification
            if track.track_id == speaker_id:
                # Find closest position
                closest_pos = None
                min_diff = float('inf')
                
                for pos in track.positions:
                    diff = abs(pos['timestamp'] - timestamp)
                    if diff < min_diff:
                        min_diff = diff
                        closest_pos = pos['center']
                
                if closest_pos:
                    return closest_pos
                
                return track.avg_position if track.avg_position else (width // 2, height // 2)
        
        # Default to center
        return (width // 2, height // 2)
    
    def _get_safe_center(self, subject_id: int, width: int, height: int) -> Tuple[int, int]:
        """Get a safe center position for a subject"""
        if subject_id in self.tracked_subjects:
            track = self.tracked_subjects[subject_id]
            if track.avg_position:
                return (int(track.avg_position[0]), int(track.avg_position[1]))
        return (width // 2, height // 2)
    
    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function for transitions"""
        if t < 0.5:
            return 2 * t * t
        return 1 - (-2 * t + 2) ** 2 / 2
    
    def _identify_speakers(self, speaker_patterns: List[Dict]) -> Dict[int, int]:
        """
        Identify which visual subject corresponds to which audio speaker
        Returns mapping of speaker_id -> track_id
        """
        mapping = {}
        
        if len(self.tracked_subjects) < 2:
            # Single subject case
            for track_id in self.tracked_subjects:
                mapping[0] = track_id
            return mapping
        
        # For multi-subject, we need to correlate speaking time with subject positions
        # This is a simplified approach - real implementation would use face-voice correlation
        
        subjects = list(self.tracked_subjects.values())
        subjects.sort(key=lambda x: x.avg_position[0])  # Sort by x position (left to right)
        
        # Assign speakers to subjects based on position
        # Speaker 0 -> leftmost subject, Speaker 1 -> rightmost subject
        for i, subject in enumerate(subjects[:2]):  # Only map first 2 subjects
            mapping[i] = subject.track_id
        
        return mapping
    
    def get_interview_tracking_points(
        self,
        video_path: str,
        target_width: int = 1080,
        target_height: int = 1920
    ) -> List[Dict]:
        """
        Get tracking points optimized for interview/talk content
        """
        # This would be called after track_subjects
        focus_points = getattr(self, '_focus_points', [])
        
        if not focus_points:
            return []
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate crop dimensions for 9:16
        crop_width = int(height * 9 / 16)
        crop_height = height
        
        if crop_width > width:
            crop_width = width
            crop_height = int(width * 16 / 9)
        
        tracking_points = []
        prev_center = (width // 2, height // 2)
        smoothing_factor = 0.3
        
        for fp in focus_points:
            target_center = fp['center']
            
            # Apply smoothing
            smooth_x = prev_center[0] * (1 - smoothing_factor) + target_center[0] * smoothing_factor
            smooth_y = prev_center[1] * (1 - smoothing_factor) + target_center[1] * smoothing_factor
            
            # Calculate crop coordinates
            crop_x = max(0, min(int(smooth_x - crop_width // 2), width - crop_width))
            crop_y = max(0, min(int(smooth_y - crop_height // 2), height - crop_height))
            
            tracking_points.append({
                'timestamp': fp['timestamp'],
                'center': (int(smooth_x), int(smooth_y)),
                'crop': (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height),
                'is_transition': fp.get('is_transition', False),
                'target_subject': fp.get('target_subject', 0)
            })
            
            prev_center = (smooth_x, smooth_y)
        
        return tracking_points


# Test
if __name__ == "__main__":
    tracker = SpeakerTracker()
    print("Speaker Tracker initialized")