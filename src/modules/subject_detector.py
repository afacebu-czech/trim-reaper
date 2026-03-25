"""
Subject Detection Module
Detects subjects (people, objects, faces) in video frames
Uses multiple detection methods for robust results
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass
from collections import defaultdict
import time

@dataclass
class Detection:
    """Single detection result"""
    frame_idx: int
    timestamp: float
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    area: int

class SubjectDetector:
    """
    Detect subjects in video frames using multiple methods
    Supports face detection, body detection, motion detection, and object detection
    """
    
    # COCO class names for YOLO
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    # Important subjects for reels
    PRIORITY_SUBJECTS = ['person', 'face', 'body', 'moving_object', 'cell phone', 'laptop', 'tv', 'car', 'dog', 'cat', 'bird']
    
    def __init__(self, model_name: str = "yolov8s", use_gpu: bool = True):
        """
        Initialize the subject detector
        
        Args:
            model_name: YOLO model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None
        self.yolo_available = False
        
        # OpenCV detectors
        self.face_cascade = None
        self.body_cascade = None
        self.upper_body_cascade = None
        self.profile_face_cascade = None
        
        # Background subtractor for motion detection
        self.bg_subtractor = None
        
        self._load_model()
        self._load_opencv_detectors()
        
        logger.info(f"Subject detector initialized - YOLO: {self.yolo_available}, OpenCV: True")
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            
            # Check for GPU
            device = 'cuda' if self.use_gpu and self._check_cuda() else 'cpu'
            
            logger.info(f"Loading YOLO model: {self.model_name} on {device}")
            self.model = YOLO(f"{self.model_name}.pt")
            self.model.to(device)
            self.yolo_available = True
            
            logger.success("YOLO model loaded successfully")
            
        except ImportError:
            logger.warning("Ultralytics not installed. Using OpenCV detection methods.")
        except Exception as e:
            logger.warning(f"YOLO not available: {str(e)}. Using OpenCV detection methods.")
    
    def _load_opencv_detectors(self):
        """Load OpenCV Haar cascades for detection"""
        try:
            # Face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Face cascade loaded")
            
            # Profile face detection
            profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            if os.path.exists(profile_path):
                self.profile_face_cascade = cv2.CascadeClassifier(profile_path)
                logger.info("Profile face cascade loaded")
            
            # Full body detection
            body_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            if os.path.exists(body_path):
                self.body_cascade = cv2.CascadeClassifier(body_path)
                logger.info("Body cascade loaded")
            
            # Upper body detection
            upper_body_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            if os.path.exists(upper_body_path):
                self.upper_body_cascade = cv2.CascadeClassifier(upper_body_path)
                logger.info("Upper body cascade loaded")
            
            # Background subtractor for motion
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True
            )
            
        except Exception as e:
            logger.warning(f"Could not load some OpenCV detectors: {str(e)}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def detect_subjects(self, video_path: str, sample_rate: int = 5) -> Dict[str, Any]:
        """
        Detect subjects throughout the video
        
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame (higher = faster but less accurate)
            
        Returns:
            Dictionary with detection results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
        
        all_detections = []
        class_counts = defaultdict(int)
        subject_timeline = []
        main_subjects = []
        
        frame_idx = 0
        prev_frame_gray = None
        prev_centers = {}
        
        # Process frames
        processed_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps
                
                # Convert to grayscale for some detections
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)  # Improve contrast
                
                detections = []
                
                # 1. YOLO Detection (if available)
                if self.yolo_available and self.model is not None:
                    yolo_dets = self._detect_yolo(frame, frame_idx, timestamp)
                    detections.extend(yolo_dets)
                
                # 2. Face Detection
                face_dets = self._detect_faces(gray, frame_idx, timestamp)
                detections.extend(face_dets)
                
                # 3. Body Detection
                body_dets = self._detect_bodies(gray, frame_idx, timestamp)
                detections.extend(body_dets)
                
                # 4. Motion Detection
                if prev_frame_gray is not None:
                    motion_dets = self._detect_motion(frame, gray, prev_frame_gray, frame_idx, timestamp)
                    detections.extend(motion_dets)
                
                # 5. Salient region detection (fallback)
                if len(detections) == 0:
                    saliency_dets = self._detect_salient_regions(frame, frame_idx, timestamp)
                    detections.extend(saliency_dets)
                
                # Remove duplicate detections (same region)
                detections = self._remove_duplicates(detections)
                
                # Store detections
                all_detections.extend(detections)
                
                # Update counts
                for det in detections:
                    class_counts[det.class_name] += 1
                
                # Track main subjects
                frame_subjects = self._get_main_subjects(detections, width, height)
                
                subject_timeline.append({
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'subjects': frame_subjects,
                    'detection_count': len(detections)
                })
                
                # Track center of main subject for smooth panning
                if frame_subjects:
                    main_subject = frame_subjects[0]
                    subject_id = f"{main_subject['class']}_{main_subject.get('track_id', 0)}"
                    
                    if subject_id in prev_centers:
                        prev_x, prev_y = prev_centers[subject_id]
                        curr_x, curr_y = main_subject['center']
                        smooth_x = prev_x * 0.7 + curr_x * 0.3
                        smooth_y = prev_y * 0.7 + curr_y * 0.3
                        main_subject['smooth_center'] = (smooth_x, smooth_y)
                    
                    prev_centers[subject_id] = main_subject['center']
                    main_subjects.append({
                        'frame': frame_idx,
                        'timestamp': timestamp,
                        **main_subject
                    })
                
                prev_frame_gray = gray.copy()
                processed_count += 1
                
                # Progress update every 30 frames
                if processed_count % 30 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    remaining = (total_frames // sample_rate - processed_count) / rate if rate > 0 else 0
                    logger.info(f"Processed {processed_count}/{total_frames // sample_rate} frames, ~{remaining:.0f}s remaining")
            
            frame_idx += 1
        
        cap.release()
        
        # Find most prominent subjects
        prominent_subjects = self._find_prominent_subjects(all_detections, width, height)
        
        # Calculate subject presence percentage
        total_processed = max(1, frame_idx // sample_rate)
        subject_presence = {
            cls: count / total_processed * 100 
            for cls, count in class_counts.items()
        }
        
        result = {
            'total_frames': frame_idx,
            'fps': fps,
            'resolution': (width, height),
            'detections': [
                {
                    'frame': d.frame_idx,
                    'timestamp': d.timestamp,
                    'class': d.class_name,
                    'confidence': d.confidence,
                    'bbox': d.bbox,
                    'center': d.center,
                    'area': d.area
                }
                for d in all_detections
            ],
            'class_counts': dict(class_counts),
            'subject_presence': subject_presence,
            'subject_timeline': subject_timeline,
            'main_subjects': main_subjects,
            'prominent_subjects': prominent_subjects,
            'sample_rate': sample_rate,
            'detection_methods': {
                'yolo': self.yolo_available,
                'opencv_faces': self.face_cascade is not None,
                'opencv_body': self.body_cascade is not None,
                'motion': True
            }
        }
        
        logger.success(f"Detection complete: {len(all_detections)} total detections from {processed_count} frames")
        logger.info(f"Class breakdown: {dict(class_counts)}")
        
        return result
    
    def _detect_yolo(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> List[Detection]:
        """Detect using YOLO"""
        detections = []
        
        if not self.yolo_available or self.model is None:
            return detections
        
        try:
            results = self.model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        if conf > 0.3:
                            class_name = self.COCO_CLASSES[cls_id] if cls_id < len(self.COCO_CLASSES) else f"class_{cls_id}"
                            
                            w = int(x2 - x1)
                            h = int(y2 - y1)
                            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            area = w * h
                            
                            detections.append(Detection(
                                frame_idx=frame_idx,
                                timestamp=timestamp,
                                class_name=class_name,
                                confidence=conf,
                                bbox=(int(x1), int(y1), w, h),
                                center=center,
                                area=area
                            ))
                            
        except Exception as e:
            logger.debug(f"YOLO detection error: {str(e)}")
        
        return detections
    
    def _detect_faces(self, gray: np.ndarray, frame_idx: int, timestamp: float) -> List[Detection]:
        """Detect faces using Haar cascades"""
        detections = []
        
        # Frontal faces
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(40, 40),
                    maxSize=(500, 500)
                )
                
                for (x, y, w, h) in faces:
                    detections.append(Detection(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        class_name='face',
                        confidence=0.85,
                        bbox=(int(x), int(y), int(w), int(h)),
                        center=(int(x + w/2), int(y + h/2)),
                        area=w * h
                    ))
            except Exception as e:
                logger.debug(f"Face detection error: {str(e)}")
        
        # Profile faces
        if self.profile_face_cascade is not None:
            try:
                profiles = self.profile_face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(40, 40)
                )
                
                for (x, y, w, h) in profiles:
                    # Check if already detected as frontal face
                    is_duplicate = False
                    for det in detections:
                        dx, dy = det.center
                        if abs(x + w/2 - dx) < w/2 and abs(y + h/2 - dy) < h/2:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        detections.append(Detection(
                            frame_idx=frame_idx,
                            timestamp=timestamp,
                            class_name='face_profile',
                            confidence=0.75,
                            bbox=(int(x), int(y), int(w), int(h)),
                            center=(int(x + w/2), int(y + h/2)),
                            area=w * h
                        ))
            except Exception as e:
                logger.debug(f"Profile face detection error: {str(e)}")
        
        return detections
    
    def _detect_bodies(self, gray: np.ndarray, frame_idx: int, timestamp: float) -> List[Detection]:
        """Detect bodies using Haar cascades"""
        detections = []
        
        # Full body
        if self.body_cascade is not None:
            try:
                bodies = self.body_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(60, 120)
                )
                
                for (x, y, w, h) in bodies:
                    detections.append(Detection(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        class_name='body',
                        confidence=0.7,
                        bbox=(int(x), int(y), int(w), int(h)),
                        center=(int(x + w/2), int(y + h/2)),
                        area=w * h
                    ))
            except Exception as e:
                logger.debug(f"Body detection error: {str(e)}")
        
        # Upper body
        if self.upper_body_cascade is not None:
            try:
                upper_bodies = self.upper_body_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(60, 80)
                )
                
                for (x, y, w, h) in upper_bodies:
                    detections.append(Detection(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        class_name='upper_body',
                        confidence=0.7,
                        bbox=(int(x), int(y), int(w), int(h)),
                        center=(int(x + w/2), int(y + h/2)),
                        area=w * h
                    ))
            except Exception as e:
                logger.debug(f"Upper body detection error: {str(e)}")
        
        return detections
    
    def _detect_motion(self, frame: np.ndarray, gray: np.ndarray, prev_gray: np.ndarray, 
                       frame_idx: int, timestamp: float) -> List[Detection]:
        """Detect motion regions"""
        detections = []
        
        try:
            # Frame difference
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Dilate to merge nearby regions
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=3)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            frame_area = frame.shape[0] * frame.shape[1]
            min_area = frame_area * 0.01  # Minimum 1% of frame
            max_area = frame_area * 0.8    # Maximum 80% of frame
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Skip very thin regions
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.2 < aspect_ratio < 5:
                        detections.append(Detection(
                            frame_idx=frame_idx,
                            timestamp=timestamp,
                            class_name='moving_object',
                            confidence=0.6,
                            bbox=(int(x), int(y), int(w), int(h)),
                            center=(int(x + w/2), int(y + h/2)),
                            area=w * h
                        ))
                        
        except Exception as e:
            logger.debug(f"Motion detection error: {str(e)}")
        
        return detections
    
    def _detect_salient_regions(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> List[Detection]:
        """Detect salient regions using image gradients and center prior"""
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize (avoid division by zero)
            max_val = grad_mag.max()
            if max_val > 0:
                grad_mag = (grad_mag / max_val * 255).astype(np.uint8)
            else:
                grad_mag = np.zeros_like(gray)
            
            # Threshold
            _, thresh = cv2.threshold(grad_mag, 50, 255, cv2.THRESH_BINARY)
            
            # Dilate
            kernel = np.ones((7, 7), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            height, width = frame.shape[:2]
            frame_area = width * height
            min_area = frame_area * 0.02
            
            # Center prior - prefer regions closer to center
            center_x, center_y = width // 2, height // 2
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate center distance
                    cx, cy = x + w/2, y + h/2
                    dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    max_dist = np.sqrt(center_x**2 + center_y**2)
                    center_score = 1 - (dist_from_center / max_dist)
                    
                    confidence = 0.4 + center_score * 0.3
                    
                    detections.append(Detection(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        class_name='salient_region',
                        confidence=confidence,
                        bbox=(int(x), int(y), int(w), int(h)),
                        center=(int(cx), int(cy)),
                        area=w * h
                    ))
            
            # If still no detections, return center region
            if not detections:
                # Use center third of frame
                x1 = width // 3
                y1 = height // 3
                x2 = 2 * width // 3
                y2 = 2 * height // 3
                
                detections.append(Detection(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    class_name='center_region',
                    confidence=0.3,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    area=(x2 - x1) * (y2 - y1)
                ))
                
        except Exception as e:
            logger.debug(f"Saliency detection error: {str(e)}")
        
        return detections
    
    def _remove_duplicates(self, detections: List[Detection]) -> List[Detection]:
        """Remove duplicate detections (overlapping regions)"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        unique = []
        
        for det in detections:
            is_duplicate = False
            x1, y1, w1, h1 = det.bbox
            
            for existing in unique:
                x2, y2, w2, h2 = existing.bbox
                
                # Calculate IoU
                xi = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                yi = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                intersection = xi * yi
                
                if intersection > 0:
                    union = w1 * h1 + w2 * h2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.5:  # More than 50% overlap
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique.append(det)
        
        return unique
    
    def _get_main_subjects(self, detections: List[Detection], frame_width: int, frame_height: int) -> List[Dict]:
        """Identify main subjects based on size and priority"""
        if not detections:
            return []
        
        frame_area = frame_width * frame_height
        center_x, center_y = frame_width // 2, frame_height // 2
        
        subjects = []
        for det in detections:
            # Calculate importance score
            size_score = det.area / frame_area * 100
            
            # Priority bonus
            priority_bonus = 20 if det.class_name in self.PRIORITY_SUBJECTS else 0
            
            # Face/person extra bonus
            face_bonus = 30 if det.class_name in ['person', 'face', 'face_profile', 'body', 'upper_body'] else 0
            
            # Confidence bonus
            conf_bonus = det.confidence * 20
            
            # Center proximity bonus (subjects closer to center are often more important)
            cx, cy = det.center
            dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            center_bonus = (1 - dist_from_center / max_dist) * 10
            
            total_score = size_score + priority_bonus + face_bonus + conf_bonus + center_bonus
            
            subjects.append({
                'class': det.class_name,
                'center': det.center,
                'bbox': det.bbox,
                'confidence': det.confidence,
                'score': total_score,
                'area': det.area
            })
        
        # Sort by score
        subjects.sort(key=lambda x: x['score'], reverse=True)
        
        return subjects
    
    def _find_prominent_subjects(self, detections: List[Detection], width: int, height: int) -> List[Dict]:
        """Find the most prominent subjects across all frames"""
        if not detections:
            return []
        
        # Group detections by class
        class_detections = defaultdict(list)
        for det in detections:
            class_detections[det.class_name].append(det)
        
        prominent = []
        frame_area = width * height
        
        for class_name, dets in class_detections.items():
            avg_area = sum(d.area for d in dets) / len(dets)
            avg_confidence = sum(d.confidence for d in dets) / len(dets)
            
            avg_x = sum(d.center[0] for d in dets) / len(dets)
            avg_y = sum(d.center[1] for d in dets) / len(dets)
            
            var_x = sum((d.center[0] - avg_x) ** 2 for d in dets) / len(dets)
            var_y = sum((d.center[1] - avg_y) ** 2 for d in dets) / len(dets)
            stability = 1 / (1 + np.sqrt(var_x + var_y) / 100)
            
            prominent.append({
                'class': class_name,
                'count': len(dets),
                'avg_area': avg_area,
                'area_percentage': avg_area / frame_area * 100,
                'avg_confidence': avg_confidence,
                'avg_center': (avg_x, avg_y),
                'stability': stability,
                'priority': class_name in self.PRIORITY_SUBJECTS
            })
        
        # Sort by area percentage and count
        prominent.sort(key=lambda x: x['area_percentage'] * x['count'], reverse=True)
        
        return prominent[:10]
    
    def get_tracking_points(self, video_path: str, target_width: int = 1080, target_height: int = 1920) -> List[Dict]:
        """Get smooth tracking points for panning/cropping"""
        result = self.detect_subjects(video_path, sample_rate=2)
        
        if not result['main_subjects']:
            return self._get_center_crop_points(result, target_width, target_height)
        
        tracking_points = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate crop dimensions for 9:16
        crop_width = int(height * 9 / 16)
        crop_height = height
        
        if crop_width > width:
            crop_width = width
            crop_height = int(width * 16 / 9)
        
        # Create subject position lookup
        subject_positions = {}
        for subject in result['main_subjects']:
            subject_positions[subject['frame']] = subject
        
        # Generate smooth crop points
        prev_center = (width // 2, height // 2)
        smoothing_factor = 0.3
        
        for frame_idx in range(total_frames):
            timestamp = frame_idx / fps
            
            # Find closest subject position
            if frame_idx in subject_positions:
                target_center = subject_positions[frame_idx]['center']
            else:
                # Find nearest frame with detection
                nearest_frame = min(subject_positions.keys(), 
                                    key=lambda f: abs(f - frame_idx)) if subject_positions else frame_idx
                if nearest_frame in subject_positions:
                    target_center = subject_positions[nearest_frame]['center']
                else:
                    target_center = prev_center
            
            # Smooth the center position
            smooth_x = prev_center[0] * (1 - smoothing_factor) + target_center[0] * smoothing_factor
            smooth_y = prev_center[1] * (1 - smoothing_factor) + target_center[1] * smoothing_factor
            
            # Calculate crop coordinates
            crop_x = max(0, min(int(smooth_x - crop_width // 2), width - crop_width))
            crop_y = max(0, min(int(smooth_y - crop_height // 2), height - crop_height))
            
            tracking_points.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'center': (int(smooth_x), int(smooth_y)),
                'crop': (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
            })
            
            prev_center = (smooth_x, smooth_y)
        
        return tracking_points
    
    def _get_center_crop_points(self, result: Dict, target_width: int, target_height: int) -> List[Dict]:
        """Generate center crop points when no subjects detected"""
        total_frames = result['total_frames']
        fps = result['fps']
        width, height = result['resolution']
        
        # Calculate crop for 9:16
        crop_width = int(height * 9 / 16)
        crop_height = height
        
        if crop_width > width:
            crop_width = width
            crop_height = int(width * 16 / 9)
        
        center_x = width // 2
        center_y = height // 2
        
        crop_x = max(0, center_x - crop_width // 2)
        crop_y = max(0, center_y - crop_height // 2)
        
        return [
            {
                'frame': i,
                'timestamp': i / fps,
                'center': (center_x, center_y),
                'crop': (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
            }
            for i in range(total_frames)
        ]


# Test
if __name__ == "__main__":
    detector = SubjectDetector(model_name="yolov8n")
    print(f"Subject Detector initialized - YOLO: {detector.yolo_available}")