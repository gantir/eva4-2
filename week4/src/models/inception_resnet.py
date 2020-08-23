import csv
import io
import json
import logging
import os

import boto3
import torch
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger()


class InceptionResnetHelper:
    def __init__(self):
        self._model_name = "inception_resnet_face_recog_v2"

    def idx_label_dict(self) -> dict:
        try:
            map = {}
            with open("data/face_recognition_clsidx_to_labels.csv", "r") as f:
                reader = csv.reader(f)
                map = {row[0]: row[1] for row in reader}
            return map
        except Exception as e:
            logger.exception(e)
            return None

    def idx_label(self, class_idx: int) -> str:
        try:
            map = self.idx_label_dict()
            return map[str(class_idx)]
        except Exception as e:
            logger.exception(e)
            return "Class Not Found"

    def load_model(self, s3_bucket: str):
        try:
            # @todo: fetch only if it doesn't exist locally
            s3 = boto3.client("s3")
            model_path = os.path.join("artifacts/models", f"{self._model_name}.pt")
            obj = s3.get_object(Bucket=s3_bucket, Key=model_path)
            logger.info("Creating Byte Stream")
            bytestream = io.BytesIO(obj["Body"].read())
            logger.info("Loading model")
            model = torch.jit.load(bytestream)
            logger.info("Model Loaded...")
            logger.info(model)
            logger.info(model.code)
            self._model = model
        except Exception as e:
            logger.exception(e)
            raise (e)

    def transform_image(self, image_bytes):
        try:
            transformations = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
            )
            image = Image.open(io.BytesIO(image_bytes))
            return transformations(image).unsqueeze(0)
        except Exception as e:
            logger.exception(e)
            raise (e)

    def get_prediction(self, image_tensor):
        if torch.cuda.is_available():
            image_tensor = image_tensor.to("cuda")
            self._model.to("cuda")
        with torch.no_grad():
            output = self._model(image_tensor).argmax().item()
            logger.info(output)
            return output, self.idx_label(output)
