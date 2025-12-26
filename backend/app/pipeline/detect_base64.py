"""
Base64 detection endpoint handler.
"""
from fastapi import HTTPException
from ultralytics import YOLO
from app.pipeline.detect import detect_objects


def handle_detect_base64(data: dict, model: YOLO) -> dict:
    """
    Handle the /detect-base64 endpoint request.
    
    Args:
        data: Request data dictionary containing 'image' key with base64 string
        model: YOLO model instance
        
    Returns:
        Dictionary with detections and count
        
    Raises:
        HTTPException: With appropriate status code for errors
    """
    try:
        # Extract base64 image data
        image_data = data.get("image")
        
        # Call the detection pipeline function
        result = detect_objects(image_data, model)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

