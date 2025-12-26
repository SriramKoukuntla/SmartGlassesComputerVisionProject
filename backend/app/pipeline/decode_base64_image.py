import base64
import cv2
import numpy as np

def decode_base64_image(image_data: str) -> np.ndarray:
    """
    Decode a base64 encoded image string into an OpenCV image.

    Args:
        image_data: Base64 encoded image string (may include data URL prefix)

    Returns:
        Decoded image as a numpy array (BGR)

    Raises:
        ValueError: If decoding fails or image is invalid
    """
    if not image_data:
        raise ValueError("No image data provided")

    try:
        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Decoded image is empty or invalid")

        return img

    except (ValueError, base64.binascii.Error) as e:
        raise ValueError("Error decoding base64 image") from e



