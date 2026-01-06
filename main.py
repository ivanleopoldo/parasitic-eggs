import io
from PIL import Image, ImageDraw, ImageFont
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src import EggClassifier, EggDetector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = EggDetector(model_path="./models/yolov8-trained.pt")
classifier = EggClassifier(
    model_path="./models/resnet50-trained.keras",
)

@app.get("/")
def root():
    return {"message": "successful"}


from PIL import Image, ImageDraw, ImageFont

def draw_boxes(image: Image.Image, results: list):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(28)

    for item in results:
        x1, y1, x2, y2 = item["bbox"]
        label = f"{item['classification']['class_name']} {item['classification']['confidence']*100:.1f}%"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")

        draw.text((x1, y1 - text_height), label, fill="white", font=font)

    return image




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

    return StreamingResponse(img_bytes, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
