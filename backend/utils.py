"""
Utility functions for image processing
"""
import cv2
import numpy as np
import base64
from typing import Optional, Tuple

def decode_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode image bytes to OpenCV image format.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        OpenCV image (numpy array) or None if decoding fails
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def decode_image_from_base64(image_data: str) -> Optional[np.ndarray]:
    """
    Decode base64 encoded image string to OpenCV image format.
    
    Args:
        image_data: Base64 encoded image string (may include data URL prefix)
        
    Returns:
        OpenCV image (numpy array) or None if decoding fails
    """
    # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
    if "," in image_data:
        image_data = image_data.split(",")[1]
    
    # Decode base64 to bytes
    try:
        image_bytes = base64.b64decode(image_data)
        return decode_image_from_bytes(image_bytes)
    except Exception:
        return None
