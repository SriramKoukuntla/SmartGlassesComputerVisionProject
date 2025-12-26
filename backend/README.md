# Smart Glasses Backend

Backend API for object detection using YOLOv11n and OCR using PaddleOCR.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Models will be automatically downloaded on first run:
   - YOLOv11n model (yolo11n.pt) for object detection
   - PaddleOCR models for text recognition (downloaded automatically)

3. Start the server:
```bash
python -m app.main
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `POST /detect-base64` - Detect objects in base64 encoded image
- `POST /ocr-base64` - Extract text from base64 encoded image

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

