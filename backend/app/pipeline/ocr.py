"""
OCR pipeline for extracting text from images using PaddleOCR.
"""
from typing import List, Dict
from paddleocr import PaddleOCR
from app.pipeline.decode import decode_base64_image


def extract_text_from_image(img, ocr_model: PaddleOCR) -> List[Dict]:
    """
    Extract text information from an image using PaddleOCR.
    
    Args:
        img: Image as numpy array
        ocr_model: PaddleOCR model instance
        
    Returns:
        List of text detection dictionaries with text, confidence, and bbox
    """
    # Run OCR
    results = ocr_model.ocr(img, cls=False)
    
    text_detections = []
    
    if results and results[0]:
        for line in results[0]:
            if line:
                # Extract bounding box coordinates
                bbox = line[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                text_info = line[1]  # (text, confidence)
                
                text = text_info[0]
                confidence = text_info[1]
                
                # Convert bbox to x1, y1, x2, y2 format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x1 = min(x_coords)
                y1 = min(y_coords)
                x2 = max(x_coords)
                y2 = max(y_coords)
                
                text_detections.append({
                    "text": text,
                    "confidence": round(confidence * 100, 2),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })
    
    return text_detections


def extract_text(image_data: str, ocr_model: PaddleOCR) -> Dict:
    """
    Process a base64 encoded image and return extracted text.
    
    Args:
        image_data: Base64 encoded image string
        ocr_model: PaddleOCR model instance
        
    Returns:
        Dictionary with text detections and count
        
    Raises:
        ValueError: If image data is invalid or cannot be decoded
    """
    if not image_data:
        raise ValueError("No image data provided")
    
    # Decode base64 to image
    img = decode_base64_image(image_data)
    
    if img is None:
        raise ValueError("Invalid image format")
    
    # Extract text from image
    text_detections = extract_text_from_image(img, ocr_model)
    
    return {
        "text_detections": text_detections,
        "count": len(text_detections),
        "full_text": " ".join([detection["text"] for detection in text_detections])
    }

