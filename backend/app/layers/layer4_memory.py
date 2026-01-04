"""Layer 4: Memory, Novelty, and Event Gating."""
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from app.config import config
from app.layers.layer2_5_risk import RiskEvent
from app.layers.layer3_reasoning import LLMResponse


@dataclass
class WorldState:
    """Maintains world state memory."""
    last_spoken_hazards: Dict[str, float] = field(default_factory=dict)  # hazard_id -> timestamp
    active_tracks: Dict[int, Dict] = field(default_factory=dict)  # track_id -> last status
    last_ocr_results: Dict[str, float] = field(default_factory=dict)  # text -> timestamp
    last_spoken_time: float = 0.0
    cooldown_timers: Dict[str, float] = field(default_factory=dict)  # event_id -> cooldown_end_time


@dataclass
class GatedEvent:
    """Event that passed gating logic."""
    event: RiskEvent
    should_speak: bool
    reason: str
    priority: float


class MemoryEventGating:
    """Decides whether to speak and what changed."""
    
    def __init__(self):
        """Initialize memory and event gating."""
        self.world_state = WorldState()
        self.cooldown_timer = config.cooldown_timer
        self.distance_delta_threshold = config.distance_delta_threshold
        self.heading_delta_threshold = config.heading_delta_threshold
    
    def _get_hazard_id(self, event: RiskEvent) -> str:
        """Generate unique ID for a hazard event."""
        if event.event_type == "object" and event.metadata:
            track_id = event.metadata.get("track_id")
            class_name = event.metadata.get("class_name", "unknown")
            if track_id is not None:
                return f"object_{track_id}_{class_name}"
            else:
                return f"object_{class_name}_{id(event)}"
        elif event.event_type == "text" and event.metadata:
            text = event.metadata.get("text", "")
            return f"text_{hash(text)}"
        elif event.event_type == "obstacle":
            return f"obstacle_{id(event)}"
        else:
            return f"unknown_{id(event)}"
    
    def _is_new_object_entering_danger_zone(self, event: RiskEvent) -> bool:
        """Check if object is newly entering danger zone."""
        if event.event_type != "object":
            return False
        
        hazard_id = self._get_hazard_id(event)
        
        # Check if we've seen this hazard recently
        if hazard_id in self.world_state.last_spoken_hazards:
            last_time = self.world_state.last_spoken_hazards[hazard_id]
            if time.time() - last_time < self.cooldown_timer:
                return False  # Already announced recently
        
        # Check if object is in danger zone
        if event.distance is not None:
            # Lower distance value = closer (for normalized depth)
            if event.distance < 0.3:  # Close threshold
                return True
        
        # Check if priority is high enough
        if event.priority > 10.0:
            return True
        
        return False
    
    def _is_object_approaching_quickly(self, event: RiskEvent) -> bool:
        """Check if object is approaching quickly."""
        if event.event_type != "object" or not event.velocity:
            return False
        
        vx, vy = event.velocity
        velocity_magnitude = (vx**2 + vy**2)**0.5
        
        # Check if velocity exceeds threshold
        if velocity_magnitude > config.approach_velocity_threshold:
            # Check if object is moving toward camera (simplified)
            # In practice, would need to check direction relative to camera
            return True
        
        return False
    
    def _is_obstacle_in_walkable_path(self, event: RiskEvent) -> bool:
        """Check if obstacle is in walkable path."""
        return event.event_type == "obstacle"
    
    def _is_new_high_confidence_ocr(self, event: RiskEvent) -> bool:
        """Check if new high-confidence OCR text detected."""
        if event.event_type != "text" or not event.metadata:
            return False
        
        text = event.metadata.get("text", "")
        if not text:
            return False
        
        # Check if we've seen this text recently
        if text in self.world_state.last_ocr_results:
            last_time = self.world_state.last_ocr_results[text]
            if time.time() - last_time < self.cooldown_timer:
                return False  # Already announced
        
        # Check confidence (assuming it's in metadata or event priority)
        if event.priority > 5.0:  # High priority text
            return True
        
        return False
    
    def _has_significant_change(self, event: RiskEvent) -> bool:
        """Check if there's been significant change in object state."""
        if event.event_type != "object" or not event.metadata:
            return False
        
        track_id = event.metadata.get("track_id")
        if track_id is None:
            return True  # New detection without track
        
        # Check if we have previous state for this track
        if track_id in self.world_state.active_tracks:
            last_state = self.world_state.active_tracks[track_id]
            
            # Check distance change
            if event.distance is not None and "distance" in last_state:
                distance_delta = abs(event.distance - last_state["distance"])
                if distance_delta > self.distance_delta_threshold:
                    return True
            
            # Check location change (heading)
            if event.location and "location" in last_state:
                x1, y1 = event.location
                x2, y2 = last_state["location"]
                # Simple heading change (could be improved)
                if abs(x1 - x2) > 50 or abs(y1 - y2) > 50:
                    return True
        
        return False
    
    def gate_events(self, events: List[RiskEvent]) -> List[GatedEvent]:
        """Gate events to determine what should be spoken.
        
        Args:
            events: List of risk events from prioritization
            
        Returns:
            List of gated events with should_speak flags
        """
        gated_events = []
        current_time = time.time()
        
        for event in events:
            should_speak = False
            reason = ""
            
            # Check various trigger conditions
            if self._is_new_object_entering_danger_zone(event):
                should_speak = True
                reason = "new_object_in_danger_zone"
            elif self._is_object_approaching_quickly(event):
                should_speak = True
                reason = "object_approaching_quickly"
            elif self._is_obstacle_in_walkable_path(event):
                should_speak = True
                reason = "obstacle_in_path"
            elif self._is_new_high_confidence_ocr(event):
                should_speak = True
                reason = "new_high_confidence_text"
            elif self._has_significant_change(event):
                should_speak = True
                reason = "significant_state_change"
            
            # Check cooldown
            hazard_id = self._get_hazard_id(event)
            if hazard_id in self.world_state.cooldown_timers:
                if current_time < self.world_state.cooldown_timers[hazard_id]:
                    should_speak = False
                    reason = "cooldown_active"
            
            gated_events.append(GatedEvent(
                event=event,
                should_speak=should_speak,
                reason=reason,
                priority=event.priority
            ))
        
        # Update world state
        self._update_world_state(gated_events)
        
        return gated_events
    
    def _update_world_state(self, gated_events: List[GatedEvent]):
        """Update world state memory with new events."""
        current_time = time.time()
        
        for gated_event in gated_events:
            event = gated_event.event
            hazard_id = self._get_hazard_id(event)
            
            if gated_event.should_speak:
                # Update last spoken time
                self.world_state.last_spoken_hazards[hazard_id] = current_time
                self.world_state.last_spoken_time = current_time
                
                # Set cooldown timer
                self.world_state.cooldown_timers[hazard_id] = current_time + self.cooldown_timer
            
            # Update active tracks
            if event.event_type == "object" and event.metadata:
                track_id = event.metadata.get("track_id")
                if track_id is not None:
                    self.world_state.active_tracks[track_id] = {
                        "location": event.location,
                        "distance": event.distance,
                        "velocity": event.velocity,
                        "class_name": event.metadata.get("class_name"),
                        "timestamp": current_time
                    }
            
            # Update OCR results
            if event.event_type == "text" and event.metadata:
                text = event.metadata.get("text", "")
                if text:
                    self.world_state.last_ocr_results[text] = current_time
        
        # Clean up old tracks (older than max_age)
        max_track_age = 5.0  # seconds
        tracks_to_remove = []
        for track_id, track_state in self.world_state.active_tracks.items():
            if current_time - track_state.get("timestamp", 0) > max_track_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.world_state.active_tracks[track_id]
        
        # Clean up old cooldown timers
        cooldowns_to_remove = []
        for event_id, cooldown_end in self.world_state.cooldown_timers.items():
            if current_time > cooldown_end:
                cooldowns_to_remove.append(event_id)
        
        for event_id in cooldowns_to_remove:
            del self.world_state.cooldown_timers[event_id]
    
    def reset(self):
        """Reset world state (useful for testing or restart)."""
        self.world_state = WorldState()

