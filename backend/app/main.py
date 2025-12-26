from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.pipeline.detect_base64 import handle_detect_base64
from app.models import yolo11n

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Smart Glasses Backend API", "status": "running"}

@app.post("/detect-base64")
async def detect_objects_base64(data: dict):
    """Process a base64 encoded image and return object detections"""
    return handle_detect_base64(data, yolo11n)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


