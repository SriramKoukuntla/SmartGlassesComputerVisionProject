"""
Detection pipeline for processing images and extracting object detections.
"""
from typing import List, Dict, Any
from ultralytics import YOLO

def runYolo(image_data: str, model: YOLO) -> Dict:
    results = model(image_data, verbose=False)
    return results

def parseYoloResult(yolo_result) -> List[Dict[str, Any]]:
    """
    Parse YOLO results object into a list of detection dictionaries.
    
    Args:
        yolo_result: YOLO Results object (can be single object or list)
        
    Returns:
        List of detection dictionaries, each containing class, confidence, and bbox
    """
    detections = []
    
    # YOLO can return a single Results object or a list
    results_list = yolo_result if isinstance(yolo_result, list) else [yolo_result]
    
    for result in results_list:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
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

def yoloResultToJSON(detections) -> Dict[str, Any]:
    """
    Convert a list of YOLO detections into a JSON-serializable dictionary.
    
    Args:
        detections: List of detection dictionaries from parseYoloResult
        
    Returns:
        Dictionary with detections list and count
    """
    return {
        "detections": detections,
    }

def executeYolo(img, model: YOLO):
    result = runYolo(img, model)
    detections = parseYoloResult(result)
    JSON = yoloResultToJSON(detections)
    return JSON