from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
import io
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI()

detection_model = YOLO("./models/best.pt")
classification_model = load_model("./models/best_resnet_model.keras")

CLASS_NAMES = [
    "Ascaris lumbricoides",
    "Trichuris trichiura",
    "Hookworm egg",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_for_resnet(image, target_size=(224, 224)):
    """
    Preprocess image for ResNet50 classification
    """
    # Resize image
    image = image.resize(target_size)

    # Convert to numpy array
    image_array = np.array(image)

    # Convert RGB to BGR if needed (ResNet typically expects BGR)
    if image_array.shape[2] == 3:
        image_array = image_array[:, :, ::-1]  # RGB to BGR

    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def classify_egg(cropped_image):
    """
    Classify cropped egg image using ResNet50
    """
    try:
        # Preprocess image for ResNet50
        processed_image = preprocess_for_resnet(cropped_image)

        # Make prediction
        predictions = classification_model.predict(processed_image, verbose=0)

        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        class_name = CLASS_NAMES[predicted_class_idx]

        return {
            "class_name": class_name,
            "class_id": int(predicted_class_idx),
            "confidence": confidence,
            "all_predictions": predictions[0].tolist(),
        }

    except Exception as e:
        print(f"Classification error: {e}")
        return {
            "class_name": "unknown",
            "class_id": -1,
            "confidence": 0.0,
            "all_predictions": [],
        }


@app.get("/")
def root():
    return {"message": "Two-Stage Egg Detection and Classification API"}


@app.post("/predict/")
async def predict_eggs(
    file: UploadFile = File(...),
    detection_confidence: float = 0.5,
    classification_confidence: float = 0.7,
):
    """
    Two-stage egg detection and classification
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await file.read()
        original_image = Image.open(io.BytesIO(image_data))
        image_width, image_height = original_image.size

        detection_results = detection_model(original_image)

        print(detection_model)

        egg_predictions = []
        successful_crops = 0

        for i, result in enumerate(detection_results):
            if result.boxes is not None and len(result.boxes) > 0:
                for j, box in enumerate(result.boxes):
                    detection_confidence_score = box.conf[0].cpu().numpy()

                    if detection_confidence_score >= detection_confidence:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        x1 = max(0, min(x1, image_width))
                        y1 = max(0, min(y1, image_height))
                        x2 = max(0, min(x2, image_width))
                        y2 = max(0, min(y2, image_height))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        padding = 10
                        x1_padded = max(0, x1 - padding)
                        y1_padded = max(0, y1 - padding)
                        x2_padded = min(image_width, x2 + padding)
                        y2_padded = min(image_height, y2 + padding)

                        try:
                            cropped_img = original_image.crop(
                                (x1_padded, y1_padded, x2_padded, y2_padded)
                            )

                            if cropped_img.size[0] == 0 or cropped_img.size[1] == 0:
                                continue

                            classification_result = classify_egg(cropped_img)

                            if (
                                classification_result["confidence"]
                                >= classification_confidence
                            ):
                                successful_crops += 1

                                egg_prediction = {
                                    "detection_id": f"{i}_{j}",
                                    "detection_confidence": float(
                                        detection_confidence_score
                                    ),
                                    "bbox": {
                                        "x1": float(x1),
                                        "y1": float(y1),
                                        "x2": float(x2),
                                        "y2": float(y2),
                                    },
                                    "bbox_normalized": {
                                        "x1": float(x1 / image_width),
                                        "y1": float(y1 / image_height),
                                        "x2": float(x2 / image_width),
                                        "y2": float(y2 / image_height),
                                    },
                                    "width": float(x2 - x1),
                                    "height": float(y2 - y1),
                                    "area": float((x2 - x1) * (y2 - y1)),
                                    "classification": classification_result,
                                }

                                egg_predictions.append(egg_prediction)

                        except Exception as crop_error:
                            print(f"Error processing crop {j}: {crop_error}")
                            continue

        egg_predictions.sort(key=lambda x: x["detection_confidence"], reverse=True)

        detection_count = len(egg_predictions)
        classification_results = [egg["classification"] for egg in egg_predictions]

        if detection_count > 0:
            avg_detection_confidence = (
                sum(egg["detection_confidence"] for egg in egg_predictions)
                / detection_count
            )
            avg_classification_confidence = (
                sum(cls["confidence"] for cls in classification_results)
                / detection_count
            )

            class_counts = {}
            for cls in classification_results:
                class_name = cls["class_name"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        else:
            avg_detection_confidence = 0
            avg_classification_confidence = 0
            class_counts = {}

        return {
            "filename": file.filename,
            "image_dimensions": {"width": image_width, "height": image_height},
            "detection_confidence_threshold": detection_confidence,
            "classification_confidence_threshold": classification_confidence,
            "total_detections": detection_count,
            "successful_classifications": successful_crops,
            "eggs": egg_predictions,
            "summary": {
                "total_eggs": detection_count,
                "average_detection_confidence": avg_detection_confidence,
                "average_classification_confidence": avg_classification_confidence,
                "classification_distribution": class_counts,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/classify-only/")
async def classify_only(file: UploadFile = File(...)):
    """
    Direct classification without detection (for testing ResNet50)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Classify directly using ResNet50
        classification_result = classify_egg(image)

        return {"filename": file.filename, "classification": classification_result}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error classifying image: {str(e)}"
        )


@app.get("/model-info/")
async def get_model_info():
    """
    Get information about loaded models
    """
    yolo_info = {
        "model_type": "YOLO",
        "task": "object_detection",
        "classes": (
            list(detection_model.names.values())
            if hasattr(detection_model, "names")
            else []
        ),
    }

    resnet_info = {
        "model_type": "ResNet50",
        "format": "Keras",
        "task": "classification",
        "classes": CLASS_NAMES,
        "input_shape": classification_model.input_shape,
        "output_shape": classification_model.output_shape,
    }

    return {"detection_model": yolo_info, "classification_model": resnet_info}


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
