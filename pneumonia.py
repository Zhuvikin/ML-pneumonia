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
from tqdm import tqdm, tqdm_notebook
from skimage import morphology
from sklearn.preprocessing import StandardScaler

cv2.__version__

# +
base_data_dir = '../data/chest_xray/'

data_dir = base_data_dir + 'prepared/'
normal_dir = data_dir + 'normal/'
pneumonia_dir = data_dir + 'pneumonia/'
bacteria_dir = pneumonia_dir + 'bacteria/'
virus_dir = pneumonia_dir + 'virus/'

for directory in [normal_dir, bacteria_dir, virus_dir]:
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


def normalize_image(original, target_width = 256):
    im_shape = (256, 256)
    original_width = original.shape[1]
    original_height = original.shape[0]
    
    width_coeff = 256 / original_width
    height_coeff = 256 / original_height
    
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
    non_empty_rows = np.where(pr.max(axis = 1) > 0)[0]
    left = min(non_empty_columns)
    right = max(non_empty_columns)
    top = min(non_empty_rows)
    bottom = max(non_empty_rows)

    left_r = left / 256
    right_r = right / 256
    top_r = top / 256
    bottom_r = bottom / 256
    
    l = floor(left / width_coeff)
    t = floor(top / height_coeff)
    w = floor((right - left) / width_coeff)
    h = floor((bottom - top) / height_coeff)
    
    cropped = original[t:t + h, l:l + w]
    resized = cv2.resize(cropped, (target_width, target_width), interpolation = cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    return equalized


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


# -

file_names_pneumonia[:10]

# +
i = 0
for path in tqdm_notebook(file_names_normal[:10]):
    img = cv2.imread(base_data_dir + path, 0)
    normalized = normalize_image(img, 256)
    cv2.imwrite(normal_dir + '{0:04d}.jpeg'.format(i), normalized)
    i += 1

i = 0
for path in tqdm_notebook(file_names_pneumonia[:20]):
    img = cv2.imread(base_data_dir + path, 0)
    normalized = normalize_image(img, 256)
    if 'bacteria' in path:
        cv2.imwrite(bacteria_dir + '{0:04d}.jpeg'.format(i), normalized)
    else:
        cv2.imwrite(virus_dir + '{0:04d}.jpeg'.format(i), normalized)
    i += 1

# +
from keras.models import load_model

model_name = '/Users/zhuvikin/workspace/lung-segmentation-2d/trained_model.hdf5'
UNet = load_model(model_name)

# +
fig = plt.figure(figsize = (17, 6))
i = 1

scaler = StandardScaler()

for problem in tqdm(file_names_pneumonia[100:118]): # problems[18:36]: 
    im_shape = (256, 256)
    img_path = base_data_dir + problem
    original = cv2.imread(img_path, 0)
    original_width = original.shape[1]
    original_height = original.shape[0]
    
    width_coeff = 256 / original_width
    height_coeff = 256 / original_height
    
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
    non_empty_rows = np.where(pr.max(axis = 1) > 0)[0]
    left = min(non_empty_columns)
    right = max(non_empty_columns)
    top = min(non_empty_rows)
    bottom = max(non_empty_rows)

    left_r = left / 256
    right_r = right / 256
    top_r = top / 256
    bottom_r = bottom / 256
    
    l = floor(left / width_coeff)
    t = floor(top / height_coeff)
    w = floor((right - left) / width_coeff)
    h = floor((bottom - top) / height_coeff)
    
    cropped = original[t:t + h, l:l + w]
    resized = cv2.resize(cropped, (256, 256), interpolation = cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)

    ax = plt.subplot(3, 6, i)
    plt.axis('off')
    plt.imshow(equalized, cmap = 'gray')

    i += 1

plt.subplots_adjust(wspace = 0.03, hspace = 0.03)
plt.show()
# -


