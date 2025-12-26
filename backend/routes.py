"""
API routes for the Smart Glasses Backend
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict
from detection import detect_objects_from_bytes, detect_objects_from_base64, process_image_input, process_image_input_base64
from terminal_stream import add_connection, remove_connection, get_recent_logs
import json

router = APIRouter()

@router.get("/")
def read_root():
    """Health check endpoint"""
    return {"message": "Smart Glasses Backend API", "status": "running"}

@router.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Process an image file and return object detections.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Dictionary with detections and count
        
    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Read image file
        contents = await file.read()
        result = detect_objects_from_bytes(contents)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/detect-base64")
async def detect_objects_base64(data: Dict):
    """
    Process a base64 encoded image and return object detections.
    
    Args:
        data: Dictionary containing "image" key with base64 encoded image
        
    Returns:
        Dictionary with detections and count
        
    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Extract base64 image data
        image_data = data.get("image")
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        result = detect_objects_from_base64(image_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/image-input")
async def image_input(file: UploadFile = File(...)):
    """
    Receive an image frame and process it for object detection.
    Results are logged to the terminal stream. No output is returned.
    
    Args:
        file: Uploaded image file
        
    Returns:
        HTTP 200 status on success
        
    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Read image file
        contents = await file.read()
        process_image_input(contents)
        return {"status": "processed"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/image-input-base64")
async def image_input_base64(data: Dict):
    """
    Receive a base64 encoded image frame and process it for object detection.
    Results are logged to the terminal stream. No output is returned.
    
    Args:
        data: Dictionary containing "image" key with base64 encoded image
        
    Returns:
        HTTP 200 status on success
        
    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Extract base64 image data
        image_data = data.get("image")
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        process_image_input_base64(image_data)
        return {"status": "processed"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.websocket("/terminal-stream")
async def terminal_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming terminal logs of what the server "sees".
    Streams object detection results, OCR results, and other processing logs.
    
    Clients receive real-time updates as JSON messages with the format:
    {
        "timestamp": "ISO timestamp",
        "type": "detection|ocr|error|...",
        "data": {...}
    }
    """
    await websocket.accept()
    add_connection(websocket)
    
    try:
        # Send recent logs to newly connected client
        recent_logs = get_recent_logs(50)  # Send last 50 logs
        for log_entry in recent_logs:
            await websocket.send_text(json.dumps(log_entry))
        
        # Keep connection alive - wait for disconnect or client messages
        while True:
            try:
                # Wait for any message from client (keeps connection alive)
                # Client can send ping messages or other commands
                # Use receive() to handle both text and disconnect events
                message = await websocket.receive()
                
                # Handle text messages (optional - client can send pings)
                if "text" in message:
                    # Echo back or handle client messages if needed
                    pass
                elif "bytes" in message:
                    # Handle binary messages if needed
                    pass
                    
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # Log the error
        from terminal_stream import add_log_entry
        add_log_entry("error", {
            "message": f"WebSocket error: {str(e)}"
        })
    finally:
        remove_connection(websocket)
