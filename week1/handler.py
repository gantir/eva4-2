import json
import logging

import utils

logger = logging.getLogger("Lambda Handler")

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
        picture = utils.get_image_from_context(event)
        picture_tensor = utils.transform_image(picture.content)
        model = utils.load_model(S3_BUCKET, "mobilenet_v2")
        prediction = utils.get_prediction(picture_tensor, model)
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps({"predicted": prediction}),
        }
    except Exception as e:
        logger.exception(e)
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": repr(e)}),
        }
