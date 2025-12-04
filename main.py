import io

import tensorflow as tf
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.losses import Loss

from src import EggClassifier, EggDetector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        focal_loss_value = weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(focal_loss_value, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


detector = EggDetector(model_path="./models/yolov8-finetuned.pt")
classifier = EggClassifier(
    model_path="./models/resnet_model_focalloss.keras",
    custom_objects={"FocalLoss": FocalLoss},
)


@app.get("/")
def root():
    return {"message": "successful"}


@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents))

    results = detector.detect_eggs(image)
    classified = classifier.classify_eggs(results)

    retval = [
        {
            "class": i["classification"]["class_name"],
            "conf": i["classification"]["confidence"],
        }
        for i in classified
    ]

    return retval


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
