# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: JKERNEL
#     language: python
#     name: jkernel
# ---

# +
import cv2
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from math import ceil, floor
from pascal_voc_writer import Writer
import shutil
from urllib import request
from imageai.Detection.Custom import DetectionModelTrainer
import tensorflow as tf
from tqdm import tqdm
from skimage import morphology
from sklearn.preprocessing import StandardScaler

cv2.__version__

# +
base_data_dir = '../data/chest_xray/'

data_dir = base_data_dir + 'prepared/'

recognition_dir = base_data_dir + 'recognition/'
recognition_train_images_dir = recognition_dir + 'train/images/'
recognition_train_annotations_dir = recognition_dir + 'train/annotations/'
recognition_test_images_dir = recognition_dir + 'test/images/'
recognition_test_annotations_dir = recognition_dir + 'test/annotations/'

for directory in [data_dir,
                  recognition_dir,
                  recognition_train_images_dir,
                  recognition_train_annotations_dir,
                  recognition_test_images_dir,
                  recognition_test_annotations_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# +
bddl = len(base_data_dir)

file_names_normal = [path[bddl:] for path in glob.glob(base_data_dir + 'train/NORMAL/*.jpeg')] + [path[bddl:] for path
                                                                                                  in glob.glob(
        '../data/chest_xray/test/NORMAL/*.jpeg')] + [path[bddl:] for path in
                                                     glob.glob('../data/chest_xray/val/NORMAL/*.jpeg')]
file_names_pneumonia = [path[bddl:] for path in glob.glob(base_data_dir + 'train/PNEUMONIA/*.jpeg')] + [path[bddl:] for
                                                                                                        path in
                                                                                                        glob.glob(
                                                                                                            '../data/chest_xray/test/PNEUMONIA/*.jpeg')] + [
                           path[bddl:] for path in glob.glob('../data/chest_xray/val/PNEUMONIA/*.jpeg')]


# +
def adjust_gamma(image, gamma = 1.0):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def adjust_contrast(image, contrast = 0):
    buf = image.copy()
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def normalize_image(image, tw = 256):
    width = image.shape[1]
    height = image.shape[0]
    nw = 512
    wc = width / nw
    nh = floor(height / wc)
    scaled_1 = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_AREA)
    equalized_1 = cv2.equalizeHist(scaled_1)

    kernel = np.ones((5, 5), np.uint8)
    adjusted_1 = adjust_gamma(equalized_1, 0.6)
    blurred_1 = cv2.GaussianBlur(adjusted_1, (13, 13), cv2.BORDER_REFLECT101)
    ret, binary_1 = cv2.threshold(blurred_1, 127, 255, cv2.THRESH_BINARY)
    dilate_1 = cv2.dilate(binary_1, kernel, iterations = 2)
    scaled_2 = cv2.resize(dilate_1, (nw, 10), interpolation = cv2.INTER_AREA)
    scaled_2 = cv2.resize(scaled_2, (nw, 1), interpolation = cv2.INTER_LANCZOS4)
    scaled_3 = cv2.resize(scaled_2, (nw, nh), interpolation = cv2.INTER_AREA)
    ret, binary_2 = cv2.threshold(scaled_3, 127, 255, cv2.THRESH_BINARY)

    white_columns = np.where((255 - binary_2).max(axis = 0) > 0)[0]
    x_start = min(white_columns)
    x_end = max(white_columns)

    cl = x_start
    cr = binary_2.shape[1] - x_end - 1

    cropped_1 = adjusted_1[:, x_start:x_end]
    equalized_2 = cv2.equalizeHist(cropped_1)

    adjusted_2 = adjust_gamma(equalized_2, 1.2)
    adjusted_3 = adjust_contrast(adjusted_2, 5)
    blurred_2 = cv2.GaussianBlur(adjusted_3, (13, 13), cv2.BORDER_REFLECT101)
    ret, binary_3 = cv2.threshold(blurred_2, 140, 255, cv2.THRESH_BINARY)

    temp = binary_3 + np.roll(binary_3, floor(nh / 4), axis = 0) \
           + np.roll(binary_3, floor(nh / 2), axis = 0) + np.roll(binary_3, floor(3 * nh / 4), axis = 0)
    temp = np.flip(temp, axis = 0) + temp
    temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
    temp = cv2.erode(temp, kernel, iterations = 9)
    temp = cv2.GaussianBlur(temp, (13, 13), cv2.BORDER_REFLECT101)
    temp = cv2.resize(temp, (equalized_2.shape[1], 3), interpolation = cv2.INTER_LANCZOS4)
    temp = cv2.resize(temp, (equalized_2.shape[1], 1), interpolation = cv2.INTER_LANCZOS4)
    temp = cv2.resize(temp, (equalized_2.shape[1], nh), interpolation = cv2.INTER_AREA)
#     ret, temp = cv2.threshold(temp, 140, 255, cv2.THRESH_BINARY)

    white_columns = np.where((255 - temp).max(axis = 0) > 0)[0]
    if len(white_columns) > 0:
        x_start = min(white_columns)
        x_end = max(white_columns)
        
        if x_start < temp.shape[1] / 2 and x_end > temp.shape[1] / 2:
            cl += x_start
            cr += temp.shape[1] - x_end - 1
            temp = temp[:, x_start:x_end]
            non_empty_columns = np.where(temp.max(axis = 0) > 0)[0]
            if len(non_empty_columns) > 0:
                x_from = min(non_empty_columns)
                x_to = max(non_empty_columns)
                cl += x_from
                cr += temp.shape[1] - x_to - 1
                normalized_1 = adjusted_2[:, x_from + 1:x_to]
            else:
                normalized_1 = adjusted_2
        else:
            normalized_1 = adjusted_2
    else:
        normalized_1 = adjusted_2

    normalized_1 = cv2.equalizeHist(normalized_1)
    twc = normalized_1.shape[1] / tw
    th = floor(normalized_1.shape[0] / twc)
    scaled_4 = cv2.resize(normalized_1, (tw, th), interpolation = cv2.INTER_AREA)

    tho = 0
    if th > tw:
        tho = floor((th - tw) / 2)
    cropped_2 = scaled_4[tho:th - tho, :]
    cropped_2 = cv2.resize(cropped_2, (tw, tw), interpolation = cv2.INTER_AREA)
    cropped_2 = cv2.equalizeHist(cropped_2)
    return cropped_2

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


# +
fig = plt.figure(figsize = (17, 6))
i = 1
for example_image in example_images:
    ROI = ROIs[i - 1]
    img_path = equalized_data_dir + file_names_pneumonia[i - 1 + 0]  # + 60
    img = cv2.imread(img_path, 0)

    ax = plt.subplot(3, 12, 2 * (i - 1) + 1)
    plt.axis('off')
    plt.imshow(img, cmap = 'gray')

    normalized = normalize_image(img, 256)
    ax = plt.subplot(3, 12, 2 * (i - 1) + 2)
    plt.axis('off')
    plt.imshow(normalized, cmap = 'magma')
    i += 1

plt.subplots_adjust(wspace = 0, hspace = 0)
plt.show()
# -


problems = ['train/NORMAL/IM-0492-0001.jpeg',
            'train/NORMAL/IM-0546-0001.jpeg',
            'train/NORMAL/NORMAL2-IM-0865-0001.jpeg',
            'train/PNEUMONIA/person26_bacteria_132.jpeg',
            'train/PNEUMONIA/person38_bacteria_195.jpeg',
            'train/PNEUMONIA/person258_bacteria_1205.jpeg',
            'train/PNEUMONIA/person272_virus_559.jpeg',
            'train/PNEUMONIA/person328_bacteria_1515.jpeg',
            'train/PNEUMONIA/person417_bacteria_1842.jpeg',
            'train/PNEUMONIA/person482_virus_984.jpeg',
            'train/PNEUMONIA/person490_bacteria_2070.jpeg',
            'train/PNEUMONIA/person565_bacteria_2348.jpeg',
            'train/PNEUMONIA/person563_bacteria_2340.jpeg',
            'train/PNEUMONIA/person571_virus_1114.jpeg',
            'train/PNEUMONIA/person605_bacteria_2468.jpeg',
            'train/PNEUMONIA/person688_virus_1282.jpeg',
            'train/PNEUMONIA/person688_virus_1281.jpeg',
            'train/PNEUMONIA/person799_virus_1431.jpeg',
            'train/PNEUMONIA/person846_bacteria_2766.jpeg',
            'train/PNEUMONIA/person902_bacteria_2827.jpeg',
            'train/PNEUMONIA/person979_virus_1654.jpeg',
            'train/PNEUMONIA/person1087_virus_1799.jpeg',
            'train/PNEUMONIA/person1162_virus_1950.jpeg',
            'train/PNEUMONIA/person1345_bacteria_3424.jpeg',
            'train/PNEUMONIA/person1395_bacteria_3544.jpeg',
            'train/PNEUMONIA/person1723_bacteria_4548.jpeg']

for path in tqdm(file_names_normal + file_names_pneumonia):
    img = cv2.imread(base_data_dir + path, 0)
    normalized = normalize_image(img, 256)

    directory = os.path.dirname(data_dir + path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(data_dir + path, normalized)

# +
from keras.models import load_model

model_name = '/Users/zhuvikin/workspace/lung-segmentation-2d/trained_model.hdf5'
UNet = load_model(model_name)

# +
fig = plt.figure(figsize = (17, 6))
i = 1
im_shape = (256, 256)
scaler = StandardScaler()

for problem in problems[18:26]: # file_names_pneumonia[0:18]: # 
    img_path = base_data_dir + problem
    original = cv2.imread(img_path, 0)
    img = cv2.resize(original, im_shape, interpolation = cv2.INTER_AREA)
    img = cv2.equalizeHist(img)
    img = adjust_gamma(img, 0.7)
    img = adjust_contrast(img, 3)
    
    img2 = img.astype(float)
    img2 = scaler.fit_transform(img2) / 3       
    img2 = np.expand_dims(img2, -1)
    inp_shape = img2.shape
    pred = UNet.predict([[img2]])[..., 0].reshape(inp_shape[:2])
    pr = pred > 0.5
    pr = remove_small_regions(pr, 0.01 * np.prod(im_shape))
    pr = pr.astype(int)
    
    non_empty_columns = np.where(pr.max(axis = 0) > 0)[0]
    left = min(non_empty_columns)
    right = max(non_empty_columns)
    top = min(non_empty_rows)
    bottom = max(non_empty_rows)
    
#     print(left, right, top, bottom)
    non_empty_rows = np.where(pr.max(axis = 1) > 0)[0]
    rect = patches.Rectangle((left, top), right - left, bottom - top,
                             linewidth=1, edgecolor='w', facecolor='none')
    
    ax = plt.subplot(3, 12, 2 * (i - 1) + 1)
    ax.add_patch(rect)
    plt.axis('off')
    plt.imshow(img, cmap = 'gray')

    ax = plt.subplot(3, 12, 2 * (i - 1) + 2)
    plt.axis('off')
    plt.imshow(pr, cmap = 'gray')
    
    i += 1

plt.subplots_adjust(wspace = 0.03, hspace = 0)
plt.show()
# -

