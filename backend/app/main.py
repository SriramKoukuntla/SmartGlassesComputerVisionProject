from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models import load_models
from app.pipeline.input_image_base64 import handle_input_image_base64

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo11n, paddleocr_en = load_models()

@app.get("/")
def read_root():
    return {"message": "Smart Glasses Backend API", "status": "running"}

@app.post("/input-image-base64")
async def input_image_base64(data: dict):
    return handle_input_image_base64(data, yolo11n, paddleocr_en)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
