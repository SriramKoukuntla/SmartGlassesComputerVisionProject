from ultralytics import YOLO
import supervision as sv


model = YOLO("yolo11n.pt")  # local file
tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def callback(frame, _):
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tid}"
        for tid in detections.tracker_id
    ]

    frame = box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections, labels)
    return frame

sv.process_video(
    source_path="people-walking.mp4",
    target_path="output.mp4",
    callback=callback
)