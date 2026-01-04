"""FastAPI endpoints for frontend integration."""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
import json
import asyncio
import base64
import time
from typing import Optional, List
from app.orchestrator import SmartGlassesOrchestrator
from app.layers.layer1_sensor import Frame
from app.layers.layer5_output import OutputMode

app = FastAPI(title="Smart Glasses API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[SmartGlassesOrchestrator] = None
frame_counter = 0


class ImageRequest(BaseModel):
    """Request model for image input."""
    image: str  # Base64 encoded image


@app.on_event("startup")
async def startup():
    """Initialize orchestrator on startup."""
    global orchestrator
    orchestrator = SmartGlassesOrchestrator()
    orchestrator.start()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global orchestrator
    if orchestrator:
        orchestrator.stop()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Smart Glasses API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "orchestrator_running": orchestrator is not None and orchestrator.is_running}


@app.post("/mode")
async def set_mode(mode: str):
    """Set output mode (navigation or description)."""
    if orchestrator:
        output_mode = OutputMode.NAVIGATION if mode == "navigation" else OutputMode.DESCRIPTION
        orchestrator.set_output_mode(output_mode)
        return {"status": "success", "mode": mode}
    return {"status": "error", "message": "Orchestrator not initialized"}


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket endpoint for video stream (deprecated - frontend handles camera)."""
    await websocket.accept()
    
    if not orchestrator:
        await websocket.close(code=1000, reason="Orchestrator not initialized")
        return
    
    # This endpoint is deprecated - frontend now handles camera input
    # Keeping for backward compatibility but it won't work without backend camera
    await websocket.send_json({
        "error": "This endpoint is deprecated. Use /input-image-base64 instead. Frontend handles camera input."
    })
    await websocket.close(code=1000, reason="Deprecated endpoint")


@app.get("/api/detections")
async def get_detections():
    """Get current detections (for polling) - deprecated.
    
    This endpoint is deprecated since frontend now handles camera input.
    Use /input-image-base64 to send images and get detections.
    """
    return {
        "error": "This endpoint is deprecated. Frontend handles camera input. Use /input-image-base64 to send images.",
        "detections": [],
        "tracks": []
    }


@app.post("/api/question")
async def ask_question(question: str):
    """Ask a question about the current scene."""
    if not orchestrator:
        return {"error": "Orchestrator not initialized"}
    
    answer = orchestrator.answer_question(question)
    return {"question": question, "answer": answer}


@app.post("/input-image-base64")
async def process_image_base64(request: ImageRequest):
    """Process image received from frontend (base64 encoded).
    
    This is the main endpoint that receives camera frames from the frontend.
    """
    global frame_counter
    
    if not orchestrator:
        return {
            "error": "Orchestrator not initialized",
            "logs": ["[ERROR] Orchestrator not initialized"]
        }
    
    try:
        # Decode base64 image
        # Format: "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        image_data = request.image
        if "," in image_data:
            # Remove data URL prefix if present
            image_data = image_data.split(",")[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {
                "error": "Failed to decode image",
                "logs": ["[ERROR] Failed to decode image from base64"]
            }
        
        # Process image through orchestrator
        frame_counter += 1
        results = orchestrator.process_image(image, frame_id=frame_counter)
        
        # Get perception output from results (already processed, no need to process again)
        perception_output = results.get("perception_output")
        if perception_output is None:
            # Fallback: process again if not in results (shouldn't happen)
            perception_output = orchestrator.perception.process_frame(image)
        
        # Format YOLO detections for frontend
        yolo_detections = []
        for det in perception_output.detections:
            yolo_detections.append({
                "bbox": {
                    "x1": float(det.bbox[0]),
                    "y1": float(det.bbox[1]),
                    "x2": float(det.bbox[2]),
                    "y2": float(det.bbox[3])
                },
                "class": det.class_name,
                "confidence": float(det.confidence * 100)  # Convert to percentage
            })
        
        # Format OCR results for frontend
        ocr_text_detections = []
        for text_region in perception_output.text_regions:
            # Convert bbox to polygon format expected by frontend
            x1, y1, x2, y2 = text_region.bbox
            polygon = [
                [float(x1), float(y1)],
                [float(x2), float(y1)],
                [float(x2), float(y2)],
                [float(x1), float(y2)]
            ]
            ocr_text_detections.append([
                text_region.text,
                float(text_region.confidence),
                polygon
            ])
        
        # Prepare response with logs
        logs = results.get("logs", [])
        logs.append(f"[{time.time():.3f}] Frame {frame_counter} processed: "
                   f"{len(yolo_detections)} objects, {len(ocr_text_detections)} text regions")
        
        return {
            "yolo_result": {
                "detections": yolo_detections
            },
            "paddleocr_result": {
                "text_detections": ocr_text_detections
            },
            "logs": logs,
            "frame_id": frame_counter,
            "processing_time": results.get("processing_time", 0),
            "fps": results.get("fps", 0)
        }
    
    except Exception as e:
        error_msg = f"[ERROR] {str(e)}"
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "logs": [error_msg]
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

