"""Utility functions and error handling for the application."""
from typing import Optional, Dict, Any, Callable
import logging
import traceback
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass


class PerceptionError(ProcessingError):
    """Exception raised during perception processing."""
    pass


class ReasoningError(ProcessingError):
    """Exception raised during reasoning processing."""
    pass


def handle_processing_error(func: Callable) -> Callable:
    """Decorator to handle processing errors gracefully.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.debug(traceback.format_exc())
            raise ProcessingError(f"Error in {func.__name__}: {str(e)}") from e
    return wrapper


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def format_bbox(bbox: tuple) -> Dict[str, float]:
    """Format bounding box tuple to dictionary.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Dictionary with formatted bbox
    """
    return {
        "x1": float(bbox[0]),
        "y1": float(bbox[1]),
        "x2": float(bbox[2]),
        "y2": float(bbox[3])
    }


def bbox_to_polygon(bbox: tuple) -> list:
    """Convert bounding box to polygon format.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        List of polygon points
    """
    x1, y1, x2, y2 = bbox
    return [
        [float(x1), float(y1)],
        [float(x2), float(y1)],
        [float(x2), float(y2)],
        [float(x1), float(y2)]
    ]

