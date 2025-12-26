"""
Base64 OCR endpoint handler.
"""
from fastapi import HTTPException
from paddleocr import PaddleOCR
from app.pipeline.ocr import extract_text


def handle_ocr_base64(data: dict, ocr_model: PaddleOCR) -> dict:
    """
    Handle the /ocr-base64 endpoint request.
    
    Args:
        data: Request data dictionary containing 'image' key with base64 string
        ocr_model: PaddleOCR model instance
        
    Returns:
        Dictionary with text detections, count, and full text
        
    Raises:
        HTTPException: With appropriate status code for errors
    """
    try:
        # Extract base64 image data
        image_data = data.get("image")
        
        # Call the OCR pipeline function
        result = extract_text(image_data, ocr_model)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

