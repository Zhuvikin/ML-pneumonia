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
                  equalized_data_dir,
                  recognition_dir,
                  recognition_train_images_dir,
                  recognition_train_annotations_dir,
                  recognition_test_images_dir,
                  recognition_test_annotations_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

pretrained_yolov3_uri = "https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5"
pretrained_yolov3_model_path = os.getcwd() + '/pretrained-yolov3.h5'
# -

if not os.path.exists(pretrained_yolov3_model_path):
    request.urlretrieve(pretrained_yolov3_uri, pretrained_yolov3_model_path)

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


# +
fig = plt.figure(figsize = (17, 6))
i = 1
for problem in problems[:18]:
    ROI = ROIs[i - 1]
    img_path = equalized_data_dir + problem + '.jpeg'
    img = cv2.imread(img_path, 0)

    ax = plt.subplot(3, 12, 2 * (i - 1) + 1)
    plt.axis('off')
    plt.imshow(img, cmap = 'gray')

    normalized = normalize_image(img, 256)
    ax = plt.subplot(3, 12, 2 * (i - 1) + 2)
    plt.axis('off')
    plt.imshow(normalized, cmap = 'magma')
    i += 1

plt.subplots_adjust(wspace = 0.03, hspace = 0)
plt.show()

# +
# for path in tqdm(file_names_normal + file_names_pneumonia):
#     img = cv2.imread(base_data_dir + path, 0)
#     normalized = normalize_image(img, 256)

#     directory = os.path.dirname(data_dir + path)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     cv2.imwrite(data_dir + path, normalized)

# +
# problems = ['train/NORMAL/IM-0492-0001',
#             'train/NORMAL/IM-0546-0001',
#             'train/NORMAL/NORMAL2-IM-0865-0001',
#             'train/PNEUMONIA/person26_bacteria_132',
#             'train/PNEUMONIA/person38_bacteria_195',
#             'train/PNEUMONIA/person258_bacteria_1205',
#             'train/PNEUMONIA/person272_virus_559',
#             'train/PNEUMONIA/person328_bacteria_1515',
#             'train/PNEUMONIA/person417_bacteria_1842',
#             'train/PNEUMONIA/person482_virus_984',
#             'train/PNEUMONIA/person490_bacteria_2070',
#             'train/PNEUMONIA/person565_bacteria_2348',
#             'train/PNEUMONIA/person563_bacteria_2340',
#             'train/PNEUMONIA/person571_virus_1114',
#             'train/PNEUMONIA/person605_bacteria_2468',
#             'train/PNEUMONIA/person688_virus_1282',
#             'train/PNEUMONIA/person688_virus_1281',
#             'train/PNEUMONIA/person799_virus_1431',
#             'train/PNEUMONIA/person846_bacteria_2766',
#             'train/PNEUMONIA/person902_bacteria_2827',
#             'train/PNEUMONIA/person979_virus_1654',
#             'train/PNEUMONIA/person1087_virus_1799',
#             'train/PNEUMONIA/person1162_virus_1950',
#             'train/PNEUMONIA/person1345_bacteria_3424',
#             'train/PNEUMONIA/person1395_bacteria_3544',
#             'train/PNEUMONIA/person1723_bacteria_4548']
# -




