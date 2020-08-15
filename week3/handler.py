try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import base64
import json
from src.libs import utils
from src.libs.logger import logger


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


def align_face(event, context):
    try:
        pictures = utils.get_image_from_event(event)
        # picture_tensor = img_net.transform_image(pictures[0].content)

        # has_human = check_human_image(picture_tensor) & check_face(picture_tensor)
        has_human = False
        if not has_human:
            return {
                "statusCode": 422,
                "headers": headers,
                "body": json.dumps({"error": "The submitted image doesn't have a human"}),
            }
        else:
            # aligned_face = get_aligned_face(picture_tensor)
            return {
                "statusCode": 200,
                "headers": headers,
            }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": repr(e)}),
        }


def face_swap(event, context):
    try:
        files = utils.get_images_from_event(event, max_files=2)

        fields = {}
        if len(files) == 2:
            for i, (file, _filename) in enumerate(files):
                fields["file" + str(i)] = (
                    "file" + str(i) + ".jpg",
                    base64.b64encode(file.content).decode("utf-8"),
                    "image/jpg",
                )

            return {"statusCode": 200, "headers": headers, "body": json.dumps(fields)}
        else:
            return {"statusCode": 400, "headers": headers, "body": "2 files couldn't be found in input"}

    except Exception as e:
        logger.exception(e)
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": repr(e)}),
        }
