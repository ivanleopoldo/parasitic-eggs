from ultralytics import YOLO
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def crop_yolo_detections(
    image_path,
    model_path="./models/best.pt",
    conf_threshold=0.7,
):
    model = YOLO(model_path)
    results = model(image_path, conf=conf_threshold)

    original_image = Image.open(image_path)

    cropped_images = []

    for i, result in enumerate(results):
        if result.boxes is not None and len(result.boxes) > 0:
            for j, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                padding = 10
                x1_padded = max(0, x1 - padding)
                y1_padded = max(0, y1 - padding)
                x2_padded = min(original_image.width, x2 + padding)
                y2_padded = min(original_image.height, y2 + padding)

                cropped_img = original_image.crop(
                    (x1_padded, y1_padded, x2_padded, y2_padded)
                )

                cropped_info = {
                    "image": cropped_img,
                    "bbox": (x1, y1, x2, y2),
                    "bbox_padded": (x1_padded, y1_padded, x2_padded, y2_padded),
                    "confidence": confidence,
                    "class_name": class_name,
                    "class_id": class_id,
                }

                cropped_images.append(cropped_info)

    return cropped_images


def classify(
    cropped_images, model_path="./models/best_resnet_model.keras", class_names=None
):
    if class_names is None:
        class_names = ["healthy_egg", "defective_egg", "unknown"]

    model = tf.keras.models.load_model(model_path)

    classification_results = []

    for i, crop_info in enumerate(cropped_images):
        try:
            processed_image = preprocess(crop_info["image"])
            predictions = model.predict(processed_image, verbose=0)

            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            class_name = class_names[predicted_class_idx]

            result = crop_info.copy()
            result["classification"] = {
                "class_name": class_name,
                "class_id": int(predicted_class_idx),
                "confidence": confidence,
                "all_predictions": predictions[0].tolist(),
            }

            classification_results.append(result)

            print(
                f"✓ Detection {i+1}: {crop_info['class_name']} -> {class_name} ({confidence:.2f})"
            )

        except Exception as e:
            print(f"✗ Error classifying detection {i+1}: {e}")
            result = crop_info.copy()
            result["classification"] = {
                "class_name": "unknown",
                "class_id": -1,
                "confidence": 0.0,
                "all_predictions": [],
            }
            classification_results.append(result)

    return classification_results


def preprocess(cropped_image, target_size=(224, 224)):
    image = cropped_image.resize(target_size)
    image_array = np.array(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    image_array = image_array[:, :, ::-1]
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def run_complete_pipeline(
    image_path,
    yolo_model_path="./models/best.pt",
    resnet_model_path="./models/best_resnet_model.keras",
    detection_conf=0.7,
    class_names=None,
):
    if class_names is None:
        class_names = ["healthy_egg", "defective_egg", "unknown"]

    cropped_images = crop_yolo_detections(
        image_path=image_path, model_path=yolo_model_path, conf_threshold=detection_conf
    )

    print(f"Found {len(cropped_images)} detections")

    if not cropped_images:
        print("No detections found")
        return []

    final_results = classify(
        cropped_images=cropped_images,
        model_path=resnet_model_path,
        class_names=class_names,
    )

    print(f"Total processed: {len(final_results)} eggs")

    if final_results:
        class_counts = {}
        for result in final_results:
            class_name = result["classification"]["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print(f"📈 Classification Summary:")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count}")

    return final_results


def visualize_medical_style(image_path, results):
    image = Image.open(image_path)

    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    plt.title(
        "Medical Parasitology: Egg Detection Results",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    color_map = {
        "Ascaris lumbricoides": "#E74C3C",
        "Hookworm egg": "#3498DB",
        "Trichuris trichiura": "#27AE60",
    }

    for i, result in enumerate(results):
        bbox = result["bbox"]
        cls = result["classification"]["class_name"]
        cls_conf = result["classification"]["confidence"]
        det_conf = result["confidence"]
        color = color_map.get(cls, "#F39C12")

        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=3,
            edgecolor=color,
            facecolor="none",
            linestyle="-",
            alpha=0.9,
        )
        plt.gca().add_patch(rect)

        short_names = {
            "Ascaris lumbricoides": "Ascaris",
            "Hookworm egg": "Hookworm",
            "Trichuris trichiura": "Trichuris",
        }
        short_class = short_names.get(cls, cls)

        plt.annotate(
            f"{short_class}\n({cls_conf:.1%})",
            xy=(bbox[0], bbox[1]),
            xytext=(bbox[0] - 10, bbox[1] - 30),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9),
            fontsize=10,
            fontweight="bold",
            color="white",
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )

        plt.text(
            bbox[0] + 5,
            bbox[1] + 5,
            f"Det: {det_conf:.2f}",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            fontsize=8,
            fontweight="bold",
            color="black",
        )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        "medical_parasite_results.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.show()


if __name__ == "__main__":
    img_path = "./samples/test2.jpg"

    class_names = ["Ascaris lumbricoides", "Hookworm egg", "Trichuris trichiura"]

    results = run_complete_pipeline(image_path=img_path, class_names=class_names)

    print(f"\n📋 Medical Parasitology Report:")
    for i, result in enumerate(results):
        print(f"Specimen {i+1}:")
        print(f"  Detection Confidence: {result['confidence']:.3f}")
        print(f"  Parasite Identification: {result['classification']['class_name']}")
        print(
            f"  Classification Confidence: {result['classification']['confidence']:.3f}"
        )
        print(
            f"  Location: [{result['bbox'][0]:.0f}, {result['bbox'][1]:.0f}, {result['bbox'][2]:.0f}, {result['bbox'][3]:.0f}]"
        )
        print("  " + "-" * 40)
        print(result)

    if results:
        visualize_medical_style(img_path, results)
