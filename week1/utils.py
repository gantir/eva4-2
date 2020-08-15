try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import base64
import logging
from requests_toolbelt.multipart import decoder, encoder

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
    try:
        filename = picture.headers[b"Content-Disposition"].decode().split(":")[1].split("-")[1]
        if 4 > len(filename):
            filename = picture.headers[b"Content-Disposition"].decode().split(":")[2].split("-")[1]
    except Exception as e:
        filename = "not-found"
        logger.exception(e)

    return filename


def get_multipartdata(file_path):

    multipartdata = encoder.MultipartEncoder(fields={"file": (file_path, open(file_path, "rb"))})
    return multipartdata


def convert_multipartdata_base64(multipartdata):
    return base64.b64encode(multipartdata.read()).decode("utf-8")
