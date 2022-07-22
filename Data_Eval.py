from PIL import Image
import os.path
import numpy as np


# Calculate Peak Signal-to-Noise Ratio for each batch of images with their HR counterparts
def PSNR(y_pred, y_true, max_val):
    psnr = 0
    for ind in range(len(y_pred)):
        mse = np.square(np.subtract(y_pred[ind], y_true[ind])).mean()
        psnr = psnr + 10 * np.log10(max_val * max_val / mse)
    return psnr / len(y_pred)


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
            im = Image.open(filepath).convert('RGB')
            im = np.array(im)
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


# Upscaling the LR image to its HR counterpart using bicubic interpolation
def Bicubic_LR_to_HR(path_for_convert, path_to_save, sf=1):
    directory = os.listdir(path_for_convert)
    for item in directory:
        filepath = os.path.join(path_for_convert, item)
        name, _ = os.path.splitext(item)
        if os.path.isfile(filepath):
            im = Image.open(filepath)
            w, h = im.size
            cim = im.resize((w * sf, h * sf), Image.BICUBIC)
            cim.save(path_to_save + name + "Scaledx" + str(sf) + ".PNG")


# Use the model to improve the resolution of the input LR image
# Input type should have the shape (None, h, w, c): [img]
# Can accept batches: [img1, img2, img3, ... ]
def LR_to_HR_transform(model, data):
    output = []
    for img in data:
        im = list()
        im.append(img)
        output.append(model.predict(np.array(im))[0])
    return np.array(output)


# Save the image tensor into png file in batches
def saveIMAGE(path_to_save, data):
    for i, im in enumerate(data):
        im = im.astype(np.uint8)
        img = Image.fromarray(im, 'RGB')
        img.save(path_to_save + "PIL_IMG_" + str(i + 1) + ".PNG")

