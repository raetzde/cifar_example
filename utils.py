import matplotlib.pyplot as plt
import numpy as np

# Helfer-Methode
import torch
import torchvision
from PIL import ImageDraw
from torchvision import transforms


def imshow(img, filename):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(filename)


def visualize_images(images, labels, classes, predicted_labels=None):
    text_color = (0, 0, 0)
    padding_size = 15
    resize_factor = 2
    all_imgs = []
    for i, (image, label) in enumerate(zip(images, labels)):
        padding = torch.ones((image.shape[0], padding_size, image.shape[-1]))
        denormailzed_img = image / 2 + 0.5
        if predicted_labels is not None:
            expanded_img = torch.cat((padding, denormailzed_img, padding), 1)
        else:
            expanded_img = torch.cat((padding, denormailzed_img), 1)

        pil_img = transforms.ToPILImage()(expanded_img)
        pil_img = pil_img.resize((pil_img.size[0] * resize_factor, pil_img.size[1] * resize_factor), resample=0)
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 10), f"{classes[label]}", text_color)

        if predicted_labels is not None:
            height = ((image.shape[1] + padding_size) * resize_factor + 10)
            draw.text((10, height), f"{classes[predicted_labels[i]]}", text_color)

        all_imgs.append(transforms.ToTensor()(pil_img))

    all_imgs = torch.stack(all_imgs)
    img_grid = torchvision.utils.make_grid(all_imgs, nrow=4)
    if predicted_labels is None:
        filename = "train_images.png"
    else:
        filename = "test_images.png"
    imshow(img_grid, filename)
