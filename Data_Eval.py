from tensorflow.python.keras.losses import mean_absolute_error
from PIL import Image
import os.path
from tensorflow import image
import numpy as np


# The L1 loss function
def l1_loss(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    loss = mae * len(y_true)
    return loss


# 2D PSNR for images
def PSNR(y_pred, y_true, max_val):
    # For array of images, (no. of image, w, h, d), calculate avg psnr
    if len(y_pred.shape) == 4:
        psnr = 0
        for ind in range(len(y_pred)):
            psnr = psnr + image.psnr(y_pred[ind], y_true[ind], max_val=max_val)
        return psnr
    elif len(y_pred.shape) == 3:
        return image.psnr(y_pred, y_true, max_val=max_val)


# re-crop all the images given a size in the folder
# Crop window is always at the center of the image
def crop(path_for_convert, path_to_save, size):
    directory = os.listdir(path_for_convert)
    for item in directory:
        filepath = os.path.join(path_for_convert, item)
        name, _ = os.path.splitext(item)
        if os.path.isfile(filepath):
            im = Image.open(filepath)
            w, h = im.size
            cim = im.crop((w/2-size[0]/2, h/2-size[1]/2, w/2+size[0]/2, h/2+size[1]/2))
            cim.save(path_to_save + name + ".PNG")


# converting image to numpy array. One folder one array
def images_to_array(path_to_dataset):
    img_array = []
    directory = os.listdir(path_to_dataset)
    for item in directory:
        filepath = os.path.join(path_to_dataset, item)
        if os.path.isfile(filepath):
            im = np.array(Image.open(filepath))
            im = im.astype('float32')
            img_array.append(im)
    return np.array(img_array)


# Umbrella function to process all the dataset into usable numpy array
# The argument "path_list", is the list containing the path to all dataset:
# The order of the path is as follow:
# path first three: x_train, x_test, x_valid
# path last half: y_train, y_test, y_valid
def makeDataset(path_list):
    data = []
    for directory in path_list:
        data.append(images_to_array(directory))
    return data

