export const API_CONFIG = {
  BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000',
  ENDPOINTS: {
    INPUT_IMAGE: '/input-image-base64', // Kept for backward compatibility
    WS_VIDEO: '/ws/video',
  },
}

export const CAMERA_CONFIG = {
  IDEAL_WIDTH: 1280,
  IDEAL_HEIGHT: 720,
  DEFAULT_REQUEST_INTERVAL: 500, // milliseconds
}

export const DETECTION_LIMITS = {
  MAX_DETECTIONS: 50,
  MAX_OCR_DETECTIONS: 50,
  MAX_LOGS: 100,
}

export const CANVAS_COLORS = {
  YOLO_BOX: '#00ff00',
  YOLO_LABEL_BG: 'rgba(0, 255, 0, 0.8)',
  YOLO_LABEL_TEXT: '#000000',
  OCR_BOX: '#0066ff',
  OCR_LABEL_BG: 'rgba(0, 102, 255, 0.8)',
  OCR_LABEL_TEXT: '#ffffff',
}

