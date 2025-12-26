"""
Model management for YOLO object detection
"""
from ultralytics import YOLO
from config import MODEL_PATH

# Global model instance
_model = None

def load_model():
    """
    Load the YOLO model and return the instance.
    Uses singleton pattern to ensure model is only loaded once.
    """
    global _model
    if _model is None:
        print(f"Loading YOLO model from {MODEL_PATH}...")
        _model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
    return _model

def get_model():
    """
    Get the loaded model instance.
    Loads the model if it hasn't been loaded yet.
    """
    if _model is None:
        return load_model()
    return _model
