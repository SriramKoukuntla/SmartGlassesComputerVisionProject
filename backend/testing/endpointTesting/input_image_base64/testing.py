import base64
import json
from pathlib import Path
import sys

# Add the backend directory to the path so we can import from app
backend_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_path))

from app.pipeline.input_image_base64 import handle_input_image_base64
from app.models import load_yolo_model, load_paddleocr_model, YOLO_MODEL_NAME

def main():
    # Get the path to the image file
    test_dir = Path(__file__).parent
    image_path = test_dir / "Unix Screenshot.png"
    
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return
    
    print("Loading image and converting to base64...")
    # Read the image file and encode it to base64
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Create the data dictionary expected by handle_input_image_base64
    data = {
        "image": image_base64
    }
    
    print("Loading YOLO model...")
    # Load YOLO model
    yolo11n = load_yolo_model(YOLO_MODEL_NAME)
    
    print("Loading PaddleOCR model...")
    # Load PaddleOCR model
    paddleocr_en = load_paddleocr_model(lang="en")
    
    print("Running handle_input_image_base64...")
    # Call the function
    try:
        result = handle_input_image_base64(data, yolo11n, paddleocr_en)
        
        print("\n" + "="*50)
        print("RESULTS:")
        print("="*50)
        print(json.dumps(result, indent=2))
        print("="*50)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

