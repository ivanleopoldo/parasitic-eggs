from ultralytics import YOLO


class EggDetector:
    def __init__(self, model_path, conf_threshold=0.7):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_eggs(self, image):
        results = self.model(image, conf=self.conf_threshold)

        cropped_images = []

        for i, result in enumerate(results):
            if result.boxes is not None and len(result.boxes) > 0:
                for j, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]

                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    padding = 10
                    x1_padded = max(0, x1 - padding)
                    y1_padded = max(0, y1 - padding)
                    x2_padded = min(image.width, x2 + padding)
                    y2_padded = min(image.height, y2 + padding)

                    cropped_img = image.crop(
                        (x1_padded, y1_padded, x2_padded, y2_padded)
                    )

                    cropped_info = {
                        "image": cropped_img,
                        "bbox": (x1, y1, x2, y2),
                        "bbox_padded": (x1_padded, y1_padded, x2_padded, y2_padded),
                        "confidence": float(confidence),
                        "class_name": class_name,
                        "class_id": class_id,
                    }

                    cropped_images.append(cropped_info)

        return cropped_images
