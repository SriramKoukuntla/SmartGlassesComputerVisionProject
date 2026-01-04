"""Layer 2: Perception Models - Fast, frame-level PyTorch-based models."""
import torch
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from app.config import config

# Object Detection (YOLO)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

# Multi-object Tracking
try:
    from collections import defaultdict
    import torch.nn as nn
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False

# OCR - PyTorch-based
try:
    import torchvision.transforms as transforms
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Depth Estimation (MiDaS)
try:
    import torch.hub
    MIDAS_AVAILABLE = True
except ImportError:
    MIDAS_AVAILABLE = False


@dataclass
class Detection:
    """Object detection result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


@dataclass
class Track:
    """Tracked object."""
    track_id: int
    detections: List[Detection]
    trajectory: List[Tuple[float, float]]  # (x, y) center points
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy)
    age: int = 0


@dataclass
class TextRegion:
    """OCR text region."""
    bbox: Tuple[float, float, float, float]
    text: str
    confidence: float


@dataclass
class DepthMap:
    """Depth estimation result."""
    depth_map: np.ndarray
    relative_depth: np.ndarray  # Normalized 0-1


@dataclass
class PerceptionOutput:
    """Combined perception model outputs."""
    detections: List[Detection]
    tracks: List[Track]
    text_regions: List[TextRegion]
    depth: Optional[DepthMap] = None
    segmentation: Optional[np.ndarray] = None  # Optional segmentation mask


class ObjectDetector:
    """2A: Object Detection using YOLO (PyTorch)."""
    
    def __init__(self, model_name: str = None):
        """Initialize YOLO detector."""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        
        self.model_name = model_name or config.yolo_model
        self.device = config.device
        self.model = YOLO(self.model_name)
        self.model.to(self.device)
        self.model.fuse()
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run object detection.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections
        """
        results = self.model(image, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.model.names[cls_id]
                
                detections.append(Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                ))
        
        return detections


class MultiObjectTracker:
    """2B: Multi-object tracking (ByteTrack or DeepSORT)."""
    
    def __init__(self, tracker_type: str = None):
        """Initialize tracker."""
        self.tracker_type = tracker_type or config.tracker_type
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.max_age = config.tracker_max_age
        self.min_hits = config.tracker_min_hits
        
        if self.tracker_type == "bytetrack":
            self._init_bytetrack()
        elif self.tracker_type == "deepsort":
            self._init_deepsort()
        else:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}")
    
    def _init_bytetrack(self):
        """Initialize ByteTrack tracker."""
        # Simplified ByteTrack implementation
        # For production, use: pip install byte-track
        self.track_thresh = 0.5
        self.high_thresh = 0.6
        self.match_thresh = 0.8
        self.frame_id = 0
    
    def _init_deepsort(self):
        """Initialize DeepSORT tracker."""
        # For production, use a proper DeepSORT implementation
        # This is a placeholder structure
        pass
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections.
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        
        # Calculate centers for tracking
        detection_centers = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            detection_centers.append(center)
        
        # Simple tracking logic (placeholder - should use proper ByteTrack/DeepSORT)
        # Match detections to existing tracks
        matched_tracks = set()
        for i, det in enumerate(detections):
            center = detection_centers[i]
            best_match = None
            best_dist = float('inf')
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                if track.trajectory:
                    last_center = track.trajectory[-1]
                    dist = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                    
                    if dist < best_dist and dist < 50:  # Threshold
                        best_dist = dist
                        best_match = track_id
            
            if best_match is not None:
                # Update existing track
                track = self.tracks[best_match]
                track.detections.append(det)
                track.trajectory.append(center)
                track.age = 0
                det.track_id = best_match
                matched_tracks.add(best_match)
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                track = Track(
                    track_id=track_id,
                    detections=[det],
                    trajectory=[center],
                    age=0
                )
                self.tracks[track_id] = track
                det.track_id = track_id
        
        # Update track ages and remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track.age += 1
                if track.age > self.max_age:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Calculate velocities
        for track in self.tracks.values():
            if len(track.trajectory) >= 2:
                recent = track.trajectory[-5:]  # Last 5 points
                if len(recent) >= 2:
                    dx = recent[-1][0] - recent[0][0]
                    dy = recent[-1][1] - recent[0][1]
                    dt = len(recent) - 1
                    track.velocity = (dx / dt, dy / dt) if dt > 0 else (0, 0)
        
        # Return active tracks
        active_tracks = [t for t in self.tracks.values() if t.age == 0]
        return active_tracks


class TextUnderstanding:
    """2C: Text understanding using EasyOCR (PyTorch-based)."""
    
    def __init__(self):
        """Initialize EasyOCR pipeline."""
        self.device = config.device
        self.confidence_threshold = config.ocr_confidence_threshold
        
        try:
            import easyocr
            # Initialize EasyOCR reader
            # 'en' for English, can add more languages: ['en', 'ch_sim', 'fr', etc.]
            # gpu=True uses GPU if available, False uses CPU
            use_gpu = (self.device == "cuda")
            self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            print("EasyOCR initialized successfully.")
        except ImportError:
            raise ImportError(
                "EasyOCR not installed. Install with: pip install easyocr"
            )
        except Exception as e:
            print(f"Warning: Failed to initialize EasyOCR: {e}")
            self.reader = None
    
    def extract_text(self, image: np.ndarray) -> List[TextRegion]:
        """Extract text from image using EasyOCR.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of text regions
        """
        if self.reader is None:
            return []
        
        try:
            # EasyOCR expects RGB, but we have BGR from OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run OCR
            # Returns: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], 'text', confidence), ...]
            results = self.reader.readtext(image_rgb)
            
            text_regions = []
            for detection in results:
                # detection format: (bbox, text, confidence)
                bbox_points, text, confidence = detection
                
                # Filter by confidence threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Convert bbox points to (x1, y1, x2, y2) format
                # bbox_points is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                x1 = min(x_coords)
                y1 = min(y_coords)
                x2 = max(x_coords)
                y2 = max(y_coords)
                
                text_regions.append(TextRegion(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    text=text,
                    confidence=float(confidence)
                ))
            
            return text_regions
        
        except Exception as e:
            print(f"Error in OCR extraction: {e}")
            return []


class DepthEstimator:
    """2D: Depth estimation using MiDaS (PyTorch)."""
    
    def __init__(self, model_name: str = None):
        """Initialize MiDaS depth estimator."""
        self.model_name = model_name or config.midas_model
        self.device = config.device
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load MiDaS model from torch.hub."""
        try:
            # Load MiDaS model
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get transform
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.model_name == "DPT_Large" or self.model_name == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
        except Exception as e:
            print(f"Warning: Could not load MiDaS model: {e}")
            print("Install with: pip install torch torchvision")
            self.model = None
    
    def estimate_depth(self, image: np.ndarray) -> Optional[DepthMap]:
        """Estimate depth map.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            DepthMap or None if model not available
        """
        if self.model is None:
            return None
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        input_batch = self.transform(img_rgb).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        
        # Convert to numpy
        depth = prediction.cpu().numpy()
        
        # Normalize to 0-1
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return DepthMap(
            depth_map=depth,
            relative_depth=depth_normalized
        )


class PerceptionModels:
    """Main perception models coordinator."""
    
    def __init__(self):
        """Initialize all perception models."""
        self.detector = ObjectDetector() if YOLO_AVAILABLE else None
        self.tracker = MultiObjectTracker() if TRACKING_AVAILABLE else None
        self.ocr = TextUnderstanding() if OCR_AVAILABLE else None
        self.depth_estimator = DepthEstimator() if MIDAS_AVAILABLE else None
    
    def process_frame(self, image: np.ndarray) -> PerceptionOutput:
        """Process a single frame through all perception models.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            PerceptionOutput with all model results
        """
        # Object detection
        detections = []
        if self.detector:
            detections = self.detector.detect(image)
        
        # Multi-object tracking
        tracks = []
        if self.tracker and detections:
            tracks = self.tracker.update(detections)
        
        # Text understanding
        text_regions = []
        if self.ocr:
            text_regions = self.ocr.extract_text(image)
        
        # Depth estimation
        depth = None
        if self.depth_estimator:
            depth = self.depth_estimator.estimate_depth(image)
        
        return PerceptionOutput(
            detections=detections,
            tracks=tracks,
            text_regions=text_regions,
            depth=depth
        )

