try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import base64
import cv2
import json
import numpy as np
from src.libs import utils
from src.libs.logger import logger
from src.models.facerec.facerec import FaceRecognition


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


def face_align(event, context):
    try:
        picture, picturename = utils.get_images_from_event(event, max_files=1)[0]
        picture_ndarray = cv2.imdecode(np.frombuffer(picture.content, np.uint8), -1)

        f = FaceRecognition()
        err, aligned_face = f.alignFace(picture_ndarray)

        fields = {"file0": ("file0", base64.b64encode(aligned_face).decode("utf-8"), "image/jpg",)}

        return {"statusCode": 200, "headers": headers, "body": json.dumps(fields)}

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


def face_mask(event, context):
    try:
        picture, picturename = utils.get_images_from_event(event, max_files=1)[0]

        n95_msk_src_img = cv2.imread("data/3M-KN95-9501-Dust-Mask_v1.jpg")
        picture_ndarray = cv2.imdecode(np.frombuffer(picture.content, np.uint8), -1)

        f = FaceRecognition()
        err, n95_msk_img = f.faceMask(n95_msk_src_img, picture_ndarray)

        fields = {"file0": ("file0", base64.b64encode(n95_msk_img).decode("utf-8"), "image/jpg",)}

        return {"statusCode": 200, "headers": headers, "body": json.dumps(fields)}

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


def face_swap(event, context):
    try:
        files = utils.get_images_from_event(event, max_files=2)
        if len(files) == 2:
            src_img_ndarray = cv2.imdecode(np.frombuffer(files[0][0].content, np.uint8), -1)
            dest_img_ndarray = cv2.imdecode(np.frombuffer(files[1][0].content, np.uint8), -1)
            f = FaceRecognition()
            err, swapped_img = f.faceSwap(src_img_ndarray, dest_img_ndarray)
            fields = {"file0": ("file0", base64.b64encode(swapped_img).decode("utf-8"), "image/jpg",)}

            return {"statusCode": 200, "headers": headers, "body": json.dumps(fields)}
        else:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": "Please pass exactly 2 files as input",
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
