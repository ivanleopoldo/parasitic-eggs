import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model


class EggClassifier:
    def __init__(
        self,
        model_path,
        custom_objects,
        class_names=["Ascaris lumbricoides", "Hookworm egg", "Trichuris trichiura"],
        input_size=(224, 224),
    ):
        self.model = load_model(model_path, custom_objects=custom_objects)
        self.class_names = class_names
        self.input_size = input_size

    def classify_eggs(self, cropped_images):
        classification_results = []

        for i, crop_info in enumerate(cropped_images):
            try:
                processed_image = self.preprocess(crop_info["image"])
                predictions = self.model.predict(processed_image, verbose=0)

                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                class_name = self.class_names[predicted_class_idx]

                result = crop_info.copy()
                result["classification"] = {
                    "class_name": class_name,
                    "class_id": int(predicted_class_idx),
                    "confidence": confidence,
                    "all_predictions": predictions[0].tolist(),
                }

                classification_results.append(result)

            except Exception:
                result = crop_info.copy()
                result["classification"] = {
                    "class_name": "unknown",
                    "class_id": -1,
                    "confidence": 0.0,
                    "all_predictions": [],
                }
                classification_results.append(result)

        return classification_results

    def preprocess(self, cropped_image):
        image = cropped_image.resize(self.input_size)

        img_array = np.array(image).astype(np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
