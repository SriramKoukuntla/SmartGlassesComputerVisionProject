# Smart Glasses Backend

Backend API for object detection using YOLOv11n.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. The YOLOv11n model will be automatically downloaded on first run (yolo11n.pt).

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
- `POST /detect` - Detect objects in uploaded image file
- `POST /detect-base64` - Detect objects in base64 encoded image

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

