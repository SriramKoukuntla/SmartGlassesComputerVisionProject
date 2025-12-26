from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pathlib import Path
from app.pipeline.detect_base64 import handle_detect_base64

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
# Model path is relative to backend directory
MODEL_PATH = Path(__file__).parent.parent / "yolo11n.pt"
print("Loading YOLOv11n model...")
model = YOLO(str(MODEL_PATH))
print("Model loaded successfully!")

@app.get("/")
def read_root():
    return {"message": "Smart Glasses Backend API", "status": "running"}

@app.post("/detect-base64")
async def detect_objects_base64(data: dict):
    """Process a base64 encoded image and return object detections"""
    return handle_detect_base64(data, model)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


