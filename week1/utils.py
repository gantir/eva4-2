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
    return picture
