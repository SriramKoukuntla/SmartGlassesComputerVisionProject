from ultralytics import YOLO
import supervision as sv
import numpy as np
import time

# Load once at startup
model = YOLO("yolov8n.pt")  # or your local "yolo11n.pt" if it exists
tracker = sv.ByteTrack()

def track_frame(frame: np.ndarray) -> dict:
    """
    Input:  frame as np.ndarray (H, W, 3) in BGR or RGB (Ultralytics handles both)
    Output: JSON-serializable dict with tracked objects + IDs
    """

    # 1) Detect
    results = model(frame, verbose=False)[0]

    # 2) Convert to Supervision detections
    detections = sv.Detections.from_ultralytics(results)

    # 3) Track (assign stable IDs across calls)
    detections = tracker.update_with_detections(detections)

    # 4) Build response (JSON-safe)
    names = results.names  # class_id -> class_name map

    objects = []
    if len(detections) > 0:
        xyxy = detections.xyxy  # (N, 4)
        conf = detections.confidence  # (N,)
        cls = detections.class_id  # (N,)
        tids = detections.tracker_id  # (N,)

        for i in range(len(detections)):
            x1, y1, x2, y2 = xyxy[i].tolist()

            class_id = int(cls[i]) if cls is not None else None
            class_name = names[class_id] if (class_id is not None and names) else None

            objects.append({
                "track_id": int(tids[i]) if tids is not None and tids[i] is not None else None,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(conf[i]) if conf is not None else None,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })

    return {
        "timestamp_ms": int(time.time() * 1000),
        "num_objects": len(objects),
        "objects": objects,
    }
