"""
Model management module for loading and managing computer vision models.
"""
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR


YOLO_MODEL_NAME = "yolo11n.pt"

def load_models():
    yolo11n = load_yolo_model(YOLO_MODEL_NAME)
    paddleocr = load_paddleocr_model(lang="en")
    return yolo11n, paddleocr

def load_yolo_model(model_name: str) -> YOLO:
    model_path = Path(__file__).parent / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading {model_name} model...")
    model = YOLO(str(model_path))
    print(f"Model {model_name} loaded successfully!")
    return model

def load_paddleocr_model(use_angle_cls: bool = False, lang: str = "en") -> PaddleOCR:
    print(f"Initializing PaddleOCR model (lang={lang})...")
    ocr = PaddleOCR(
        lang="en",                 # language
        use_angle_cls=True,        # handles rotated text
        det=True,                  # text detection
        rec=True,                  # text recognition
        cls=True,                  # angle classifier
        use_gpu=True,              # set False if no GPU
        det_db_score_mode="slow",  # more accurate box filtering
        show_log=False             # silence paddle logs
    )
    print("PaddleOCR model initialized successfully!")
    return ocr_model





# Load default models
yolo11n = load_yolo_model("yolo11n.pt")
paddleocr_en = load_paddleocr_model(lang="en")

__all__ = ["load_yolo_model", "load_paddleocr_model", "yolo11n", "paddleocr_en"]

