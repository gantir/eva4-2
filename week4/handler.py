try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import base64
import json
from src.libs import utils
from src.libs.logger import logger
import os
from src.models.inception_resnet import InceptionResnetHelper
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests


headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}
S3_BUCKET = "eva4-p2"


def hello(event, context):
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event,
    }
    print("Testing hello of eva4p2")

    response = {"statusCode": 200, "body": json.dumps(body)}

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """


def _face_align(picture, picture_name):
    try:
        face_align_url = "https://gh1xz0gzpj.execute-api.ap-south-1.amazonaws.com/dev/face_align"
        # face_align_url = "http://localhost:3000/dev/face_align"
        m = MultipartEncoder(fields={"files[0]": (picture_name, picture.content, "image/jpeg")})

        response = requests.post(
            face_align_url, data=base64.b64encode(m.read()).decode("utf-8"), headers={"content-type": m.content_type}
        )
        if response.status_code == requests.codes.ok:
            return json.loads(response.text)["file0"][1]
        response.raise_for_status()
    except Exception as e:
        logger.warning(e)
        raise e


def face_rec(event, context):
    try:
        picture, picture_name = utils.get_images_from_event(event, max_files=1)[0]
        aligned_face = base64.b64decode(_face_align(picture, picture_name))
        # aligned_face = base64.b64decode(_face_align(picture, picture_name).encode("utf-8"))
        inception_resnet = InceptionResnetHelper()
        inception_resnet.load_model(S3_BUCKET)
        picture_tensor = inception_resnet.transform_image(aligned_face)
        prediction_idx, prediction_label = inception_resnet.get_prediction(picture_tensor)
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps({"file": picture_name, "predicted": prediction_label}),
        }
    except ValueError as ve:
        logger.exception(ve)
        return {
            "statusCode": 422,
            "headers": headers,
            "body": json.dumps({"error": repr(ve)}),
        }
    except Exception as e:
        logger.exception(e)
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": repr(e)}),
        }
