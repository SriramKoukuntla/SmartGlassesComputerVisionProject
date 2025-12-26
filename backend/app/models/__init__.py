"""
Model management module for loading and managing computer vision models.
"""
from pathlib import Path
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR



YOLO_MODEL_NAME = "yolo11n.pt"
MIDAS_MODEL_NAME = "DPT_Hybrid"


def load_models():
    yolo11n = load_yolo_model(YOLO_MODEL_NAME)
    paddleocr = load_paddleocr_model(lang="en")
    midas = load_MiDaS_model()
    return yolo11n, paddleocr, midas

def load_yolo_model(model_name: str) -> YOLO:
    model_path = Path(__file__).parent / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading {model_name} model...")
    model = YOLO(str(model_path))
    print(f"Model {model_name} loaded successfully!")
    return model

def load_paddleocr_model(use_angle_cls: bool = False, lang: str = "en") -> PaddleOCR:
    print(f"Initializing PaddleOCR model ")
    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True)
    print("PaddleOCR model initialized successfully!")
    return ocr

def load_MiDaS_model() -> MiDaS:
    print(f"Loading MiDaS model...")
    midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_NAME)
    print("MiDaS model loaded successfully!")
    if torch.cuda.is_available():
        midas.to(torch.device("cuda"))
        print("MiDaS model moved to GPU")
    else:
        midas.to(torch.device("cpu"))
        print("MiDaS model moved to CPU")
    midas.eval()
    return midas