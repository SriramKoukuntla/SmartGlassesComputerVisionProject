# Smart Glasses Backend

## Architecture

This backend implements a 5-layer architecture for smart glasses computer vision:

### Layer 1 — Sensor Ingest
- Raw video stream (camera frames + timestamps)
- Optional: audio stream (mic)

### Layer 2 — Perception Models (PyTorch-based)
- **2A: Object Detection** - YOLO (Ultralytics, PyTorch)
- **2B: Multi-object Tracking** - ByteTrack or DeepSORT (PyTorch)
- **2C: Text Understanding** - PyTorch OCR pipeline (CRAFT/DBNet + CRNN/Transformer)
- **2D: Depth/Geometry** - MiDaS (PyTorch) for relative depth estimation

### Layer 2.5 — Risk & Prioritization
- Computes risk scores based on proximity, velocity, class weights, and walkable path
- Maintains priority queue of top N events

### Layer 3 — Scene Reasoning + Language (LLM)
- Builds structured scene graph from perception outputs
- Generates natural language descriptions using LLM (OpenAI, Anthropic, or local)
- Answers user questions about the scene

### Layer 4 — Memory, Novelty, and Event Gating
- Maintains world state memory (last spoken hazards, active tracks, OCR results)
- Triggers speech only on significant changes (new objects, approaching objects, obstacles, new text)
- Implements cooldown timers and change thresholds

### Layer 5 — Output & Interaction
- TTS with priority and interruptibility
- Two modes: Navigation (short commands) or Description (rich context)
- Optional: haptics/tones for urgent alerts

## Installation

### 1. Create and activate virtual environment

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install PyTorch with CUDA

**Important**: Install PyTorch with CUDA support for GPU acceleration. Visit https://pytorch.org/get-started/locally/ and select your system configuration.

Example for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install other dependencies

```bash
pip install -r requirements.txt
```

### 4. Install OCR (EasyOCR)

EasyOCR is a PyTorch-based OCR library that handles both text detection and recognition:

pip install easyocrEasyOCR will automatically download the required models on first use. The models are PyTorch-based and will use GPU if available (when PyTorch is installed with CUDA support).

**Note**: First-time initialization may take a few minutes as it downloads the models. Subsequent runs will be faster.

Alternatively, you can use other PyTorch OCR libraries that bundle detection + recognition.

### 5. Configure LLM (optional)

For Layer 3 (Scene Reasoning), set environment variables:

```bash
# For OpenAI
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export LLM_API_KEY=your_api_key_here

# For Anthropic
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-haiku-20240307
export LLM_API_KEY=your_api_key_here
```

If no LLM is configured, the system will use rule-based fallback descriptions.

## Usage

### Command Line

Run the main application:

```bash
python -m app.main
```

Options:
- `--mode {navigation,description}`: Output mode (default: navigation)
- `--camera ID`: Camera device ID (default: 0)
- `--debug`: Enable debug output

### API Server

Run the FastAPI server for frontend integration:

```bash
python -m app.api
```

Or using uvicorn directly:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

API Endpoints:
- `GET /`: Root endpoint
- `GET /health`: Health check
- `POST /mode`: Set output mode
- `GET /api/detections`: Get current detections (polling)
- `POST /api/question`: Ask question about scene
- `WS /ws/video`: WebSocket for video stream

## Configuration

Edit `app/config.py` to customize:
- Camera settings (resolution, FPS)
- Model selection (YOLO model, OCR models, MiDaS model)
- Risk thresholds
- LLM settings
- TTS settings
- Memory/cooldown timers

## Performance Notes

- **GPU Support**: Essential for real-time performance. Ensure PyTorch is installed with CUDA.
- **Model Selection**: Use smaller models (e.g., YOLOv8n) for faster inference on edge devices.
- **Frame Rate**: Adjust processing frequency in `orchestrator.py` to balance accuracy vs. speed.
- **LLM Calls**: LLM descriptions are generated every 10 frames by default to reduce API costs.

## Troubleshooting

1. **Camera not working**: Check camera permissions and device ID
2. **Slow performance**: Verify GPU/CUDA installation and model sizes
3. **OCR not working**: Install and configure OCR models (see Installation step 4)
4. **LLM errors**: Check API keys and network connectivity
5. **TTS not working**: Install pyttsx3 or configure alternative TTS provider
