from app.pipeline.decode_base64_image import decode_base64_image
from app.pipeline.YoloObjectDetection import executeYolo
from app.pipeline.PaddleOcrTextRecognition import executePaddleOcr
from fastapi import HTTPException
from ultralytics import YOLO
from paddleocr import PaddleOCR

def handle_input_image_base64(data: dict, yolo11n: YOLO, paddleocr_en: PaddleOCR):
    try:
        image_data = data.get("image")
        img = decode_base64_image(image_data)

        yolo_JSON = executeYolo(img, yolo11n)
        paddleocr_JSON = executePaddleOcr(img, paddleocr_en)
        midas_JSON = executeMiDaS(img, midas)

        
        return {
            "yolo_result": yolo_JSON,
            "paddleocr_result": paddleocr_JSON,
            "midas_result": midas_JSON
        }
 
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_details = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_details)

