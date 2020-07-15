try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import base64
import io
import logging
import os

import boto3
import torch
import torchvision.transforms as transforms
from PIL import Image
from requests_toolbelt.multipart import decoder

# Retrieve the logger instance
logger = logging.getLogger()

models_filename = {
    "mobilenet_v2": "mobilenet_v2-b0353104.pth",
    "resnet34": "resnet34-333f7ec4.pth",
}


def load_model(s3_bucket: str, model_name: str):
    try:
        # s3 = boto3.session.Session(profile_name='eva4p2').client("s3") # noqa
        s3 = boto3.client("s3")
        model_path = os.path.join(
            "artifacts/models", models_filename[model_name]
        )
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


def get_image_from_event(event):
    content_type_header = event["headers"]["content-type"]
    body = base64.b64decode(event["body"])
    logger.info("Body Loaded")

    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    return picture


def get_prediction(image_tensor, model):
    if torch.cuda.is_available():
        image_tensor = image_tensor.to("cuda")
        model.to("cuda")
    with torch.no_grad():
        output = model(image_tensor).argmax().item()
        logger.info(output)
        return output
