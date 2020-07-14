import json
import torch
from PIL import Image
from torchvision import transforms
import base64
from requests_toolbelt.multipart import decoder
import logging
logger = logging.getLogger("Lambda Handler")

def hello(event, context):
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event,
    }
    print('Testing hello of eva4p2')

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
def mobile_net_v2(event, context):
    try:
        content_type_header =event["headers"]["content-type"]
        body = base64.b64decode(event["body"])
        logger.info("Body Loaded")

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        picture_tensor = _transform_image(picture.content)
        prediction = _get_prediction(image_tensor = picture_tensor)

    except expression as identifier:
        pass