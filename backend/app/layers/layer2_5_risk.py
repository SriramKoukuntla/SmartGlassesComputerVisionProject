"""Layer 2.5: Risk & Prioritization - Cheap logic for computing risk scores."""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from app.config import config
from app.layers.layer2_perception import Detection, Track, TextRegion, DepthMap, PerceptionOutput


@dataclass
class RiskEvent:
    """Risk event with priority score."""
    event_type: str  # "object", "text", "obstacle"
    priority: float  # Higher = more urgent
    description: str
    location: Optional[Tuple[float, float]] = None
    distance: Optional[float] = None
    velocity: Optional[Tuple[float, float]] = None
    metadata: Dict = None


class RiskPrioritization:
    """Computes risk scores and maintains priority queue of events."""
    
    # Class weights (higher = more dangerous)
    CLASS_WEIGHTS = {
        "car": 10.0,
        "truck": 10.0,
        "bus": 10.0,
        "motorcycle": 8.0,
        "bicycle": 6.0,
        "person": 5.0,
        "dog": 4.0,
        "cat": 3.0,
        "chair": 2.0,
        "couch": 2.0,
        "potted plant": 1.0,
    }
    
    def __init__(self):
        """Initialize risk prioritization."""
        self.max_items = config.max_priority_items
        self.danger_zone_distance = config.danger_zone_distance
        self.approach_velocity_threshold = config.approach_velocity_threshold
    
    def compute_risk_score(
        self,
        detection: Detection,
        track: Optional[Track] = None,
        depth: Optional[DepthMap] = None,
        walkable_path: Optional[np.ndarray] = None
    ) -> float:
        """Compute risk score for a detection.
        
        Args:
            detection: Object detection
            track: Associated track (if available)
            depth: Depth map (if available)
            walkable_path: Segmentation mask of walkable area (if available)
            
        Returns:
            Risk score (higher = more dangerous)
        """
        score = 0.0
        
        # Base class weight
        class_name = detection.class_name.lower()
        class_weight = self.CLASS_WEIGHTS.get(class_name, 1.0)
        score += class_weight * detection.confidence
        
        # Proximity / depth
        if depth is not None:
            x1, y1, x2, y2 = detection.bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Get depth at object center
            if 0 <= center_y < depth.relative_depth.shape[0] and 0 <= center_x < depth.relative_depth.shape[1]:
                relative_depth = depth.relative_depth[center_y, center_x]
                # Lower depth value = closer = higher risk
                proximity_score = (1.0 - relative_depth) * 5.0
                score += proximity_score
        
        # Approach velocity
        if track and track.velocity:
            vx, vy = track.velocity
            velocity_magnitude = np.sqrt(vx**2 + vy**2)
            if velocity_magnitude > self.approach_velocity_threshold:
                # Object is moving quickly
                score += velocity_magnitude * 2.0
        
        # Location relative to walkable path
        if walkable_path is not None:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            # Check if object overlaps with walkable path
            obj_mask = np.zeros(walkable_path.shape, dtype=bool)
            obj_mask[y1:y2, x1:x2] = True
            overlap = np.sum(obj_mask & walkable_path) / np.sum(obj_mask)
            
            if overlap > 0.3:  # Object is in walkable path
                score += 8.0
        
        # Confidence boost
        score *= (0.5 + detection.confidence * 0.5)
        
        return score
    
    def compute_text_risk_score(self, text_region: TextRegion) -> float:
        """Compute risk score for text region.
        
        Args:
            text_region: OCR text region
            
        Returns:
            Risk score
        """
        # Text is generally lower priority than objects
        base_score = 2.0
        
        # Boost for high confidence
        score = base_score * text_region.confidence
        
        # Check for important keywords
        text_lower = text_region.text.lower()
        important_keywords = ["stop", "danger", "warning", "caution", "exit", "entrance"]
        if any(keyword in text_lower for keyword in important_keywords):
            score += 5.0
        
        return score
    
    def prioritize_events(
        self,
        perception_output: PerceptionOutput,
        walkable_path: Optional[np.ndarray] = None
    ) -> List[RiskEvent]:
        """Compute risk scores and return top N priority events.
        
        Args:
            perception_output: Output from perception models
            walkable_path: Optional segmentation mask of walkable area
            
        Returns:
            List of risk events sorted by priority (highest first)
        """
        events = []
        
        # Process detections/tracks
        track_dict = {track.track_id: track for track in perception_output.tracks}
        
        for detection in perception_output.detections:
            track = track_dict.get(detection.track_id) if detection.track_id else None
            
            risk_score = self.compute_risk_score(
                detection=detection,
                track=track,
                depth=perception_output.depth,
                walkable_path=walkable_path
            )
            
            # Calculate location and distance
            x1, y1, x2, y2 = detection.bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            distance = None
            
            if perception_output.depth:
                center_x, center_y = int(center[0]), int(center[1])
                if 0 <= center_y < perception_output.depth.relative_depth.shape[0]:
                    if 0 <= center_x < perception_output.depth.relative_depth.shape[1]:
                        distance = float(perception_output.depth.relative_depth[center_y, center_x])
            
            velocity = track.velocity if track else None
            
            events.append(RiskEvent(
                event_type="object",
                priority=risk_score,
                description=f"{detection.class_name} (confidence: {detection.confidence:.2f})",
                location=center,
                distance=distance,
                velocity=velocity,
                metadata={
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "track_id": detection.track_id,
                    "bbox": detection.bbox
                }
            ))
        
        # Process text regions
        for text_region in perception_output.text_regions:
            risk_score = self.compute_text_risk_score(text_region)
            
            x1, y1, x2, y2 = text_region.bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            events.append(RiskEvent(
                event_type="text",
                priority=risk_score,
                description=f"Text: {text_region.text}",
                location=center,
                metadata={
                    "text": text_region.text,
                    "bbox": text_region.bbox
                }
            ))
        
        # Check for obstacles in walkable path
        if walkable_path is not None and perception_output.detections:
            for detection in perception_output.detections:
                x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
                obj_mask = np.zeros(walkable_path.shape, dtype=bool)
                if y1 < walkable_path.shape[0] and x1 < walkable_path.shape[1]:
                    obj_mask[max(0, y1):min(walkable_path.shape[0], y2),
                            max(0, x1):min(walkable_path.shape[1], x2)] = True
                    overlap = np.sum(obj_mask & walkable_path) / (np.sum(obj_mask) + 1e-8)
                    
                    if overlap > 0.5:  # Significant overlap with walkable path
                        events.append(RiskEvent(
                            event_type="obstacle",
                            priority=15.0,  # High priority for obstacles
                            description=f"Obstacle in path: {detection.class_name}",
                            location=((x1 + x2) / 2, (y1 + y2) / 2),
                            metadata={"class_name": detection.class_name}
                        ))
        
        # Sort by priority and return top N
        events.sort(key=lambda e: e.priority, reverse=True)
        return events[:self.max_items]

