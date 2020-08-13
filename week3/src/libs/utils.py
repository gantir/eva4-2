try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import base64
from requests_toolbelt.multipart import decoder, encoder

from src.libs.logger import logger


def _decode_multipart_file(multi_part_file, content_type_header):
    body = base64.b64decode(multi_part_file)
    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    filename = get_picture_filename(picture).replace('"', "")
    return picture, filename


def get_images_from_event(event, max_files=1):
    content_type_header = event["headers"]["content-type"]
    pics = event["files"]
    pic_details = []
    if str == type(pics):
        pic_details.append(_decode_multipart_file(pics, content_type_header))
    else:
        for i, pic in enumerate(pics):
            pic_details.append(_decode_multipart_file(pic, content_type_header))

    return pic_details[0:max_files]


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
