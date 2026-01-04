"""FastAPI endpoints for frontend integration."""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import json
import asyncio
from typing import Optional
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
    """WebSocket endpoint for video stream."""
    await websocket.accept()
    
    if not orchestrator:
        await websocket.close(code=1000, reason="Orchestrator not initialized")
        return
    
    try:
        while orchestrator.is_running:
            # Read frame
            frame = orchestrator.sensor.read_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            # Process frame
            results = orchestrator.process_frame(frame)
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Send frame and results
            await websocket.send_bytes(frame_bytes)
            await websocket.send_json({
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "results": results
            })
            
            await asyncio.sleep(0.033)  # ~30 FPS
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


@app.get("/api/detections")
async def get_detections():
    """Get current detections (for polling)."""
    if not orchestrator or not orchestrator.is_running:
        return {"detections": [], "tracks": []}
    
    # Read current frame
    frame = orchestrator.sensor.read_frame()
    if frame is None:
        return {"detections": [], "tracks": []}
    
    # Process frame
    perception_output = orchestrator.perception.process_frame(frame.image)
    
    # Format detections
    detections = []
    for det in perception_output.detections:
        detections.append({
            "bbox": det.bbox,
            "confidence": det.confidence,
            "class_id": det.class_id,
            "class_name": det.class_name,
            "track_id": det.track_id
        })
    
    # Format tracks
    tracks = []
    for track in perception_output.tracks:
        tracks.append({
            "track_id": track.track_id,
            "trajectory": track.trajectory,
            "velocity": track.velocity,
            "age": track.age
        })
    
    return {
        "detections": detections,
        "tracks": tracks,
        "text_regions": [{"bbox": tr.bbox, "text": tr.text, "confidence": tr.confidence} 
                        for tr in perception_output.text_regions],
        "has_depth": perception_output.depth is not None
    }


@app.post("/api/question")
async def ask_question(question: str):
    """Ask a question about the current scene."""
    if not orchestrator:
        return {"error": "Orchestrator not initialized"}
    
    answer = orchestrator.answer_question(question)
    return {"question": question, "answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

