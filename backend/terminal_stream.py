"""
Terminal stream management for logging detection results
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Set
from collections import deque

# Store active WebSocket connections
_active_connections: Set = set()

# Store recent log entries (circular buffer)
_log_buffer = deque(maxlen=1000)  # Keep last 1000 log entries


def add_log_entry(log_type: str, data: Dict):
    """
    Add a log entry to the buffer and queue it for broadcasting.
    
    Args:
        log_type: Type of log (e.g., "detection", "ocr", "error")
        data: Log data dictionary
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": log_type,
        "data": data
    }
    
    # Add to buffer
    _log_buffer.append(log_entry)
    
    # Queue message for broadcasting (will be sent to active connections)
    if _active_connections:
        try:
            # Try to get the running event loop and schedule the broadcast
            loop = asyncio.get_running_loop()
            asyncio.create_task(_broadcast_message(json.dumps(log_entry)))
        except RuntimeError:
            # No running event loop - this shouldn't happen in FastAPI context
            # but we'll handle it gracefully
            pass

async def _broadcast_message(message: str):
    """
    Broadcast a message to all active WebSocket connections.
    
    Args:
        message: JSON string message to broadcast
    """
    disconnected = set()
    for connection in _active_connections:
        try:
            await connection.send_text(message)
        except Exception:
            disconnected.add(connection)
    
    # Remove disconnected connections
    _active_connections -= disconnected

def add_connection(websocket):
    """Add a WebSocket connection to the active connections set."""
    _active_connections.add(websocket)

def remove_connection(websocket):
    """Remove a WebSocket connection from the active connections set."""
    _active_connections.discard(websocket)

def get_recent_logs(count: int = 100) -> List[Dict]:
    """
    Get recent log entries from the buffer.
    
    Args:
        count: Number of recent logs to retrieve
        
    Returns:
        List of log entries
    """
    return list(_log_buffer)[-count:]
