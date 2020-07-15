try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import base64
import io
import logging
import os
import json

import boto3
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet, mobilenet

from PIL import Image
from requests_toolbelt.multipart import decoder

# Retrieve the logger instance
logger = logging.getLogger()


def get_image_from_event(event):
    content_type_header = event["headers"]["content-type"]
    body = base64.b64decode(event["body"])
    logger.info("Body Loaded")

    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    return picture


model_func_map = {
    "mobilenet_v2": mobilenet.mobilenet_v2,
    "resnet34": resnet.resnet34,
}


def imagenet_classidx_to_labels(class_idx: int) -> str:
    try:
        with open("data/imagenet1000_clsidx_to_labels.json", "r") as f:
            map = json.loads(f.read())
            return map[str(class_idx)]
    except Exception as e:
        logger.exception(e)
        return "Class Not Found"


def save_pretrained_model(model_name):
    try:
        model_func = model_func_map[model_name]
        model = model_func(pretrained=True)
        model.eval()
        # trace model with dummy input
        traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
        traced_model.save(f"./models/{model_name}.pt")
    except Exception as e:
        logger.exception(e)


def load_model(s3_bucket: str, model_name: str):
    try:
        # s3 = boto3.session.Session(profile_name='eva4p2').client("s3") # noqa
        s3 = boto3.client("s3")
        model_path = os.path.join("artifacts/models", f"{model_name}.pt")
        obj = s3.get_object(Bucket=s3_bucket, Key=model_path)
        logger.info("Creating Byte Stream")
        bytestream = io.BytesIO(obj["Body"].read())
        logger.info("Loading model")
        model = torch.jit.load(bytestream)
        logger.info("Model Loaded...")
        return model
    except Exception as e:
        logger.exception(e)
        raise (e)


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        logger.exception(e)
        raise (e)
