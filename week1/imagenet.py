import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import mobilenet
from torchvision.models import resnet


class ImageNetHelper:
    def __init__(self, model_name):
        _model_func_map = {
            "mobilenet_v2": mobilenet.mobilenet_v2,
            "resnet34": resnet.resnet34,
        }
        self._model_name = model_name
        self._model_func = _model_func_map[model_name]

    def imagenet1000_classidx_label_map(self) -> dict:
        try:
            with open("data/imagenet1000_clsidx_to_labels.json", "r") as f:
                map = json.loads(f.read())
                return map
        except Exception as e:
            logger.exception(e)
            return None

    def imagenet1000_classidx_to_label(self, class_idx: int) -> str:
        try:
            map = json.loads(self.imagenet1000_classidx_label_map())
            return map[str(class_idx)]
        except Exception as e:
            logger.exception(e)
            return "Class Not Found"

    def save_pretrained_model(self):
        try:
            model = self._model_func(pretrained=True)
            model.eval()
            # trace model with dummy input
            traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
            traced_model.save(f"./models/{self._model_name}.pt")
        except Exception as e:
            logger.exception(e)

    def load_model(self, s3_bucket: str):
        try:
            # s3 = boto3.session.Session(profile_name='eva4p2').client("s3") # noqa
            s3 = boto3.client("s3")
            model_path = os.path.join(
                "artifacts/models", f"{self._model_name}.pt"
            )
            obj = s3.get_object(Bucket=s3_bucket, Key=model_path)
            logger.info("Creating Byte Stream")
            bytestream = io.BytesIO(obj["Body"].read())
            logger.info("Loading model")
            model = torch.jit.load(bytestream)
            logger.info("Model Loaded...")
            logger.info(model)
            logger.info(model.code)
            return model
        except Exception as e:
            logger.exception(e)
            raise (e)

    def transform_image(self, image_bytes):
        try:
            transformations = transforms.Compose(
                [
                    transforms.Resize(255),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            image = Image.open(io.BytesIO(image_bytes))
            return transformations(image).unsqueeze(0)
        except Exception as e:
            logger.exception(e)
            raise (e)

    def get_prediction(self, image_tensor, model):
        if torch.cuda.is_available():
            image_tensor = image_tensor.to("cuda")
            model.to("cuda")
        with torch.no_grad():
            output = model(image_tensor).argmax().item()
            logger.info(output)
            return output
