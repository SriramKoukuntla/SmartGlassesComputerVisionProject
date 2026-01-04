"""Configuration settings for the application."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration."""
    # Camera settings
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    
    # Model settings
    yolo_model: str = "yolov8n.pt"  # Ultralytics YOLO
    device: str = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
    # Tracking settings
    tracker_type: str = "bytetrack"  # or "deepsort"
    tracker_max_age: int = 30
    tracker_min_hits: int = 3
    
    # OCR settings
    ocr_detection_model: str = "easyocr"  # Using EasyOCR (handles both detection and recognition)
    ocr_recognition_model: str = "easyocr"  # Using EasyOCR
    ocr_confidence_threshold: float = 0.5
    
    # Depth settings
    midas_model: str = "DPT_Large"  # or "MiDaS_small"
    
    # Risk & Prioritization
    max_priority_items: int = 5
    danger_zone_distance: float = 2.0  # meters
    approach_velocity_threshold: float = 0.5  # m/s
    
    # LLM settings
    llm_provider: Optional[str] = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, local
    llm_model: Optional[str] = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")
    llm_temperature: float = 0.3
    
    # Memory & Event Gating
    cooldown_timer: float = 2.0  # seconds
    distance_delta_threshold: float = 0.3  # meters
    heading_delta_threshold: float = 15.0  # degrees
    
    # TTS settings
    tts_provider: str = "pyttsx3"  # or "gtts", "azure"
    tts_rate: int = 150
    tts_volume: float = 0.8
    navigation_mode: bool = True  # True for short commands, False for descriptions
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False


config = Config()

