"""
Image decoding utilities for processing base64 encoded images.
"""
import cv2
import numpy as np
import base64
from typing import Optional


def decode_base64_image(image_data: str) -> Optional[np.ndarray]:
    """
    Decode a base64 encoded image string into a numpy array.
    
    Args:
        image_data: Base64 encoded image string (may include data URL prefix)
        
    Returns:
        Decoded image as numpy array, or None if decoding fails
    """
    try:
        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception:
        return None

