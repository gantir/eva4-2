try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import base64
import logging

from requests_toolbelt.multipart import decoder

# Retrieve the logger instance
logger = logging.getLogger()


def get_image_from_event(event):
    content_type_header = event["headers"]["content-type"]
    body = base64.b64decode(event["body"])
    logger.info("Body Loaded")

    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    filename = get_picture_filename(picture).replace('"', "")
    return picture, filename


def get_picture_filename(picture):
    filename = (
        picture.headers[b"Content-Disposition"]
        .decode()
        .split(":")[1]
        .split("-")[1]
    )
    if 4 > len(filename):
        filename = (
            picture.headers[b"Content-Disposition"]
            .decode()
            .split(":")[2]
            .split("-")[1]
        )

    return filename
