try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import json
import logging

import utils

# Initialize you log configuration using the base class
logging.basicConfig(level=logging.INFO)
# Retrieve the logger instance
logger = logging.getLogger()

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


def classify_image(event, context):
    try:
        picture = utils.get_image_from_event(event)
        picture_tensor = utils.transform_image(picture.content)
        model = utils.load_model(S3_BUCKET, "mobilenet_v2")
        prediction_idx = utils.get_prediction(picture_tensor, model)
        prediction_label = utils.imagenet_classidx_to_labels(prediction_idx)
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(
                {"predicted": (prediction_idx, prediction_label)}
            ),
        }
    except Exception as e:
        logger.exception(e)
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": repr(e)}),
        }
