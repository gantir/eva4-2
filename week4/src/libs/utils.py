try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import base64
import os
from requests_toolbelt.multipart import decoder, encoder

from src.libs.logger import logger


def get_images_from_event(event, max_files=1):
    try:
        pic_details = []

        content_type_key = "Content-Type"
        if event["headers"].get(content_type_key.lower()):
            content_type_key = content_type_key.lower()
        content_type_header = event["headers"][content_type_key]

        body = base64.b64decode(event["body"])
        if type(event["body"]) is str:
            event["body"] = bytes(event["body"], "utf-8")

        pictures = decoder.MultipartDecoder(body, content_type_header)
        for part in pictures.parts:
            filename = get_picture_filename(part).replace('"', "")
            pic_details.append((part, filename))

        logger.info(f"len: {len(pic_details)}")

        return pic_details[0:max_files]

    except Exception as e:
        logger.exception(e)
        raise e


def get_picture_filename(picture):
    try:
        header = picture.headers[b"Content-Disposition"].decode("utf-8")
        key = ""
        filename = ""

        if "filename" in header:
            key = "filename"
        elif "name" in header:
            key = "name"

        if key:
            filename = [x.strip() for x in header.split(";") if key in x][0].split("=")[1].strip('"')

    except Exception as e:
        filename = "not-found"
        logger.exception(e)

    return filename


def get_multipartdata(file_path):

    multipartdata = encoder.MultipartEncoder(
        fields={"file": (os.path.basename(file_path), open(file_path, "rb").read())}
    )
    return multipartdata


def convert_multipartdata_base64(multipartdata):
    return base64.b64encode(multipartdata.read()).decode("utf-8")
