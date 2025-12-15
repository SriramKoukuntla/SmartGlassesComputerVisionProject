from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
from typing import List, Dict
import base64
import easyocr

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv11n model
print("Loading YOLOv11n model...")
model = YOLO('yolo11n.pt')
print("Model loaded successfully!")

# Initialize EasyOCR reader
print("Loading EasyOCR model...")
ocr_reader = easyocr.Reader(['en'], gpu=False)  # Use GPU if available, otherwise CPU
print("EasyOCR model loaded successfully!")

@app.get("/")
def read_root():
    return {"message": "Smart Glasses Backend API", "status": "running"}

def perform_ocr(img):
    """
    Perform OCR on the image and return text detections
    """
    try:
        # Run OCR
        ocr_results = ocr_reader.readtext(img)
        
        text_detections = []
        for (bbox, text, confidence) in ocr_results:
            # bbox is a list of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            # Convert to x1, y1, x2, y2 format
            # Convert numpy types to Python native types
            x_coords = [float(point[0]) for point in bbox]
            y_coords = [float(point[1]) for point in bbox]
            x1 = float(min(x_coords))
            y1 = float(min(y_coords))
            x2 = float(max(x_coords))
            y2 = float(max(y_coords))
            
            # Convert confidence to Python float
            conf_value = float(confidence) * 100
            
            text_detections.append({
                "text": str(text),  # Ensure text is a string
                "confidence": round(conf_value, 2),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })
        
        return text_detections
    except Exception as e:
        print(f"OCR error: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Process an image and return object detections and text recognition
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Run YOLOv11n detection
        results = model(img, verbose=False)
        
        # Extract detection information
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
        
        # Perform OCR for text recognition
        text_detections = perform_ocr(img)
        
        return {
            "detections": detections,
            "text_detections": text_detections,
            "count": len(detections),
            "text_count": len(text_detections)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-base64")
async def detect_objects_base64(data: dict):
    """
    Process a base64 encoded image and return object detections and text recognition
    """
    try:
        # Extract base64 image data
        image_data = data.get("image")
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Run YOLOv11n detection
        results = model(img, verbose=False)
        
        # Extract detection information
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
        
        # Perform OCR for text recognition
        text_detections = perform_ocr(img)
        
        return {
            "detections": detections,
            "text_detections": text_detections,
            "count": len(detections),
            "text_count": len(text_detections)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

