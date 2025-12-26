"""
Model management module for loading and managing computer vision models.
"""
from pathlib import Path
from ultralytics import YOLO


def load_yolo_model(model_name: str = "yolo11n.pt") -> YOLO:
    """
    Load a YOLO model from the models directory.
    
    Args:
        model_name: Name of the model file (default: "yolo11n.pt")
        
    Returns:
        Loaded YOLO model instance
    """
    # Model path is relative to the models directory
    model_path = Path(__file__).parent / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading {model_name} model...")
    model = YOLO(str(model_path))
    print(f"Model {model_name} loaded successfully!")
    
    return model


# Load default YOLOv11n model
yolo11n = load_yolo_model("yolo11n.pt")

__all__ = ["load_yolo_model", "yolo11n"]

