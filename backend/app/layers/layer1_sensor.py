"""Layer 1: Sensor Ingest - Raw video stream with timestamps."""
import cv2
import time
from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass
from app.config import config


@dataclass
class Frame:
    """Frame with timestamp."""
    image: np.ndarray
    timestamp: float
    frame_id: int


class SensorIngest:
    """Handles raw video stream ingestion with timestamps."""
    
    def __init__(self, camera_id: Optional[int] = None):
        """Initialize sensor ingest.
        
        Args:
            camera_id: Camera device ID (default from config)
        """
        self.camera_id = camera_id or config.camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_id = 0
        self.is_running = False
        
    def start(self) -> bool:
        """Start video capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, config.fps)
            
            self.is_running = True
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def read_frame(self) -> Optional[Frame]:
        """Read next frame with timestamp.
        
        Returns:
            Frame object or None if failed
        """
        if not self.is_running or self.cap is None:
            return None
        
        ret, image = self.cap.read()
        if not ret:
            return None
        
        timestamp = time.time()
        frame = Frame(
            image=image,
            timestamp=timestamp,
            frame_id=self.frame_id
        )
        self.frame_id += 1
        return frame
    
    def stop(self):
        """Stop video capture."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

