"""
Detection pipeline for processing images and extracting object detections.
"""
from typing import List, Dict
from ultralytics import YOLO
from app.pipeline.decode import decode_base64_image


def extract_detections(results, model: YOLO) -> List[Dict]:
    """
    Extract detection information from YOLO results.
    
    Args:
        results: YOLO model results
        model: YOLO model instance (for class names)
        
    Returns:
        List of detection dictionaries with class, confidence, and bbox
    """
    detections = []
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


def detect_objects(image_data: str, model: YOLO) -> Dict:
    """
    Process a base64 encoded image and return object detections.
    
    Args:
        image_data: Base64 encoded image string
        model: YOLO model instance
        
    Returns:
        Dictionary with detections and count
        
    Raises:
        ValueError: If image data is invalid or cannot be decoded
    """
    if not image_data:
        raise ValueError("No image data provided")
    
    # Decode base64 to image
    img = decode_base64_image(image_data)
    
    if img is None:
        raise ValueError("Invalid image format")
    
    # Run YOLOv11n detection
    results = model(img, verbose=False)
    
    # Extract detection information
    detections = extract_detections(results, model)
    
    return {
        "detections": detections,
        "count": len(detections)
    }

