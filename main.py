import io
from PIL import Image, ImageDraw, ImageFont
import uvicorn
from huggingface_hub import hf_hub_download
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from dotenv import load_dotenv
load_dotenv('.env')

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import base64

from src import EggClassifier, EggDetector

HF_TOKEN = os.environ['HF_TOKEN']

resnet50_model_path = hf_hub_download(
    repo_id="milktt1/resnet50-parasitic-egg-detection",
    filename="resnet50-trained.keras",
    local_dir="models",
    token=HF_TOKEN
)

yolov8_model_path = hf_hub_download(repo_id="milktt1/yolov8-parasitic-detection", local_dir="models", filename="yolov8-trained.pt", token=HF_TOKEN)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


detector = EggDetector(model_path=yolov8_model_path)
classifier = EggClassifier(
    model_path=resnet50_model_path
)

def draw_boxes(image: Image.Image, results: list, labels=True):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(28)

    for idx, item in enumerate(results):
        x1, y1, x2, y2 = item["bbox"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        if labels: 
            label = f"{idx} {item['classification']['class_name']} {item['classification']['confidence']*100:.1f}%"
        else:
            label = ""

        bbox = draw.textbbox((0, 0), label, font=font)

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")

        draw.text((x1, y1 - text_height), label, fill="white", font=font)

    return image

def serialize_results(results):
    serialized = []
    for item in results:
        serialized.append({
            "bbox": item["bbox"], 
            "classification": {
                "class_name": item["classification"]["class_name"],
                "confidence": float(item["classification"]["confidence"])
            } if "classification" in item else None
        })
    return serialized


@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = detector.detect_eggs(image)
    classified = classifier.classify_eggs(results)

    image_with_boxes = draw_boxes(image, classified)

    img_bytes = io.BytesIO()
    image_with_boxes.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    encoded_img = base64.b64encode(img_bytes.read()).decode("utf-8")

    return {
        "image": encoded_img,
        "results": serialize_results(classified)
    }

@app.post("/detect")
async def detect(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = detector.detect_eggs(image)
    image_with_boxes = draw_boxes(image, results, labels=False)

    img_bytes = io.BytesIO()
    image_with_boxes.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    encoded_img = base64.b64encode(img_bytes.read()).decode("utf-8")

    return {
        "image": encoded_img,
        "results": serialize_results(results)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
