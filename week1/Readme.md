# Week 1

## Imagenet classification

This contains a simple AWS lambda deployment of a Pytorch model. The steps are as follows for classifying (imgaenet) the image:

Offline:

1. Take the model from PyTorch and save it as Torchscript model.
2. Upload the Torchscript model to S3.

On Request:
1. Read the multipart image into bytes
2. Convert the image into tensor
3. Load the Torchscript model
4. Classify the image using the model and the image tensor
5. Map the class index to human readable label name

API:
Not pasting the API link here as it can be crawled by bots and will incurr cost. Please mail me if you want the endpoint for imagenet classification

References:
https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
