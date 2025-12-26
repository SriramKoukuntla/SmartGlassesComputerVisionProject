"""
Object detection processing logic
"""
import numpy as np
from typing import List, Dict
from models import get_model
from utils import decode_image_from_bytes, decode_image_from_base64
from terminal_stream import add_log_entry

def process_detections(results) -> List[Dict]:
    """
    Extract detection information from YOLO results.
    
    Args:
        results: YOLO model detection results
        
    Returns:
        List of detection dictionaries with class, confidence, and bbox
    """
    detections = []
    model = get_model()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name and confidence
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                "class": class_name,
                "confidence": round(confidence * 100, 2),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })
    
    return detections

def detect_objects_in_image(image: np.ndarray) -> Dict:
    """
    Run object detection on an image.
    
    Args:
        image: OpenCV image (numpy array)
        
    Returns:
        Dictionary with detections and count
    """
    model = get_model()
    results = model(image, verbose=False)
    detections = process_detections(results)
    
    return {
        "detections": detections,
        "count": len(detections)
    }

def detect_objects_from_bytes(image_bytes: bytes) -> Dict:
    """
    Detect objects in an image from raw bytes.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Dictionary with detections and count
        
    Raises:
        ValueError: If image cannot be decoded
    """
    img = decode_image_from_bytes(image_bytes)
    if img is None:
        raise ValueError("Invalid image format")
    
    return detect_objects_in_image(img)

def detect_objects_from_base64(image_data: str) -> Dict:
    """
    Detect objects in an image from base64 encoded string.
    
    Args:
        image_data: Base64 encoded image string
        
    Returns:
        Dictionary with detections and count
        
    Raises:
        ValueError: If image cannot be decoded
    """
    img = decode_image_from_base64(image_data)
    if img is None:
        raise ValueError("Invalid image format")
    
    return detect_objects_in_image(img)

def process_image_input(image_bytes: bytes):
    """
    Process an image input frame and log detection results to terminal stream.
    This function is used for the /image-input endpoint which doesn't return data.
    
    Args:
        image_bytes: Raw image bytes
        
    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        img = decode_image_from_bytes(image_bytes)
        if img is None:
            raise ValueError("Invalid image format")
        
        # Run detection
        result = detect_objects_in_image(img)
        
        # Log detection results to terminal stream
        if result["count"] > 0:
            log_data = {
                "message": f"Detected {result['count']} object(s)",
                "detections": result["detections"],
                "count": result["count"]
            }
            add_log_entry("detection", log_data)
        else:
            add_log_entry("detection", {
                "message": "No objects detected",
                "detections": [],
                "count": 0
            })
            
    except Exception as e:
        # Log errors to terminal stream
        add_log_entry("error", {
            "message": f"Error processing image: {str(e)}"
        })
        raise

def process_image_input_base64(image_data: str):
    """
    Process a base64 encoded image input frame and log detection results to terminal stream.
    This function is used for the /image-input endpoint which doesn't return data.
    
    Args:
        image_data: Base64 encoded image string
        
    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        img = decode_image_from_base64(image_data)
        if img is None:
            raise ValueError("Invalid image format")
        
        # Run detection
        result = detect_objects_in_image(img)
        
        # Log detection results to terminal stream
        if result["count"] > 0:
            log_data = {
                "message": f"Detected {result['count']} object(s)",
                "detections": result["detections"],
                "count": result["count"]
            }
            add_log_entry("detection", log_data)
        else:
            add_log_entry("detection", {
                "message": "No objects detected",
                "detections": [],
                "count": 0
            })
            
    except Exception as e:
        # Log errors to terminal stream
        add_log_entry("error", {
            "message": f"Error processing image: {str(e)}"
        })
        raise
