"""
Configuration settings for the Smart Glasses Backend
"""
from typing import List

# CORS Configuration
CORS_ORIGINS: List[str] = [
    "http://localhost:5173",
    "http://localhost:3000"
]

# Model Configuration
MODEL_PATH: str = "yolo11n.pt"

# Server Configuration
HOST: str = "0.0.0.0"
PORT: int = 8000
