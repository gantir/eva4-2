import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_acc_loss(train_acc_series, test_acc_series, train_loss_series, test_loss_series):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training & Validation")

    for line in train_acc_series.items():
        axs[0, 0].plot(line[1], label=line[0])
    axs[0, 0].set_title("Training Accuracy")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Accuracy")
    axs[0, 0].legend()

    for line in train_loss_series.items():
        axs[0, 1].plot(line[1], label=line[0])
    axs[0, 1].set_title("Training Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()

    for line in test_acc_series.items():
        axs[1, 0].plot(line[1], label=line[0])
    axs[1, 0].set_title("Validation Accuracy")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Accuracy")
    axs[1, 0].legend()

    for line in test_loss_series.items():
        axs[1, 1].plot(line[1], label=line[0])
    axs[1, 1].set_title("Validation Loss")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Loss")
    axs[1, 1].legend()


def plot_accuracy(train_history, val_history):
    _ = plt.plot(train_history)
    _ = plt.plot(val_history)
    _ = plt.title("Model Accuracy")
    _ = plt.ylabel("Accuracy")
    _ = plt.xlabel("Epoch")
    _ = plt.legend(["train", "val"], loc="upper left")
    _ = plt.show()


def plot_loss(train_history, val_history):
    _ = plt.plot(train_history)
    _ = plt.plot(val_history)
    _ = plt.title("Model Loss")
    _ = plt.ylabel("Loss")
    _ = plt.xlabel("Epoch")
    _ = plt.legend(["train", "val"], loc="upper left")
    _ = plt.show()


def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]):
    single_img = False
    if tensor.ndimension() == 3:
        single_img = True
        tensor = tensor[None, :, :, :]

    if not tensor.ndimension() == 4:
        raise TypeError("tensor should be 4D")

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    ret = tensor.mul(std).add(mean)
    return ret[0] if single_img else ret


def imshow_torch(image):
    di = denormalize(image)
    np_image = di.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))


def plot_images(img_data, classes, img_name):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Images")

    num_images = len(img_data)
    for index in range(1, num_images + 1):
        img = denormalize(img_data[index - 1]["img"])
        # img = img_data[index - 1]["img"]
        plt.subplot(5, 5, index)
        plt.axis("off")
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title(
            "Predicted: %s\nActual: %s"
            % (classes[img_data[index - 1]["pred"]], classes[img_data[index - 1]["target"]],)
        )

    plt.tight_layout()
