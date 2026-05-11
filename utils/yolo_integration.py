from ultralytics import YOLO

class YOLOPlantDetector:
    def __init__(self, model_name='yolov8n.pt'):
        self.model = YOLO(model_name)
        self.conf_threshold = 0.3

    def detect_plant_count(self, image_path):
        results = self.model(image_path, conf=self.conf_threshold)
        detections = results[0].boxes
        if detections is None:
            return 50
        count = len(detections)
        density = min(200, max(50, count * 2.5))
        return int(density)

    def extract_feature_from_image(self, image_path):
        return self.detect_plant_count(image_path)