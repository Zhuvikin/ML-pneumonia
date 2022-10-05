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
equalized_data_dir = data_dir + 'equalized/'

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
# for path in tqdm(file_names_normal + file_names_pneumonia):
#     img = cv2.imread(base_data_dir + path, 0)
#     equ = cv2.equalizeHist(img)
#     dir = os.path.dirname(equalized_data_dir + path)
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     cv2.imwrite(equalized_data_dir + path, equ)
# -

file_names_normal[1557]

# +
example_images = [
    file_names_normal[1557],
    file_names_normal[1434],
    file_names_normal[1582],
    file_names_normal[1579],
    file_names_normal[1577],
    file_names_normal[1578],
    file_names_normal[1576],
    file_names_normal[1580],
    file_names_normal[1575],
    file_names_normal[1581],
    file_names_pneumonia[4268],
    file_names_pneumonia[4270],
    file_names_pneumonia[4269],
    file_names_pneumonia[4271],
    file_names_pneumonia[4265],
    file_names_pneumonia[4266],
    file_names_pneumonia[4267],
    file_names_pneumonia[4272],
]

examples = [cv2.imread(equalized_data_dir + file, 0) for file in example_images]

fig = plt.figure(figsize = (16, 6))

i = 1
for example in examples:
    plt.subplot(3, 6, i)
    plt.axis('off')
    ax = plt.imshow(example)
    i += 1

plt.show()


# +
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def adjust_contrast(image, contrast = 0):
    buf = image.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


# +
ROIs = [
    [272, 41, 1631, 1317],
    [301, 20, 1810, 1409],
    [224, 174, 1532, 1174],
    [79, 157, 1070, 891],
    [232, 182, 1568, 1162],
    [149, 170, 1112, 807],
    [121, 246, 1057, 906],
    [189, 140, 1221, 916],
    [121, 116, 1521, 1090],
    [154, 175, 1368, 1091],
    [107, 45, 856, 533],
    [133, 64, 904, 582],
    [69, 61, 1019, 604],
    [144, 71, 1286, 830],
    [86, 64, 1121, 778],
    [100, 48, 831, 585],
    [101, 101, 1046, 849],
    [254, 131, 1381, 886],
]

fig = plt.figure(figsize = (16, 6))

def normalize_image(img):
    width = img.shape[1]
    height = img.shape[0]
    nw = 512
    wc = width / nw
    nh = floor(height / wc)
    img2 = cv2.resize(img, (nw, nh), interpolation = cv2.INTER_AREA)
    img3 = cv2.equalizeHist(img2)
    kernel = np.ones((5,5), np.uint8)
    img4 = adjust_gamma(img3, 0.6)
    img4a = cv2.GaussianBlur(img4, (13, 13), cv2.BORDER_REFLECT101)
    ret, img4b = cv2.threshold(img4a, 127, 255, cv2.THRESH_BINARY)
    img4c = cv2.dilate(img4b, kernel, iterations=2)
    img4d = cv2.resize(img4c, (nw, 10), interpolation = cv2.INTER_AREA)
    img4d = cv2.resize(img4d, (nw, 1), interpolation = cv2.INTER_LANCZOS4)
    img4e = cv2.resize(img4d, (nw, nh), interpolation = cv2.INTER_AREA)
    ret, img4f = cv2.threshold(img4e, 127, 255, cv2.THRESH_BINARY)
    
    white_columns = np.where((255 - img4f).max(axis=0) > 0)[0]
    x_start = min(white_columns)
    x_end = max(white_columns)
    
    cl = x_start
    cr = img4f.shape[1] - x_end - 1
    
    img4g = img4[:,x_start:x_end]
    img4h = cv2.equalizeHist(img4g)
        
    img5 = adjust_gamma(img4h, 1.2)
    img5a = adjust_contrast(img5, 5)    
    img5b = cv2.GaussianBlur(img5a, (13, 13), cv2.BORDER_REFLECT101)
    ret, img5c = cv2.threshold(img5b, 140, 255, cv2.THRESH_BINARY)
    
    img5d = img5c + np.roll(img5c, floor(nh / 4), axis=0) + np.roll(img5c, floor(nh / 2), axis=0) + np.roll(img5c, floor(3 * nh / 4), axis=0)
    img5d = np.flip(img5d, axis=0) + img5d
    img5d = cv2.morphologyEx(img5d, cv2.MORPH_CLOSE, kernel)
    img5d = cv2.erode(img5d, kernel, iterations=9)
    img5d = cv2.GaussianBlur(img5d, (13, 13), cv2.BORDER_REFLECT101)
    img5d = cv2.resize(img5d, (img4h.shape[1], 3), interpolation = cv2.INTER_LANCZOS4)
    img5d = cv2.resize(img5d, (img4h.shape[1], 1), interpolation = cv2.INTER_LANCZOS4)
    img5d = cv2.resize(img5d, (img4h.shape[1], nh), interpolation = cv2.INTER_AREA)
    ret, img5d = cv2.threshold(img5d, 140, 255, cv2.THRESH_BINARY)
    
    white_columns = np.where((255 - img5d).max(axis=0) > 0)[0]
    
    if len(white_columns) > 0:
        x_start = min(white_columns)
        x_end = max(white_columns)
        cl += x_start
        cr += img5d.shape[1] - x_end - 1
        img5d = img5d[:,x_start:x_end]

        rows, cols = img5d.shape    
        non_empty_columns = np.where(img5d.max(axis=0) > 0)[0]
        if len(non_empty_columns) > 0:
            x_from = min(non_empty_columns)
            x_to = max(non_empty_columns)
            cl += x_from
            cr += img5d.shape[1] - x_to - 1
            img10 = img5[:, x_from + 1:x_to]
        else:
            img10 = img5
    else:
        img10 = img5
            
    img10 = cv2.equalizeHist(img10)
    
    tw = 256
    twc = img10.shape[1] / tw
    th = floor(img10.shape[0] / twc)
    img11 = cv2.resize(img10, (tw, th), interpolation = cv2.INTER_AREA)
    
    tho = 0
    if th > tw:
        tho = floor((th - tw) / 2)
    img12 = img11[tho:th - tho,:]
    img12 = cv2.resize(img12, (tw, tw), interpolation = cv2.INTER_AREA)
    img12 = cv2.equalizeHist(img12)
    return img12

i = 1
for example_image in example_images:
    ROI = ROIs[i - 1]
    img_path = equalized_data_dir + example_images[i - 1 + 0] # + 60
    img = cv2.imread(img_path, 0)
    normalized = normalize_image(img)
    ax = plt.subplot(3, 6, i)
    plt.axis('off')
    plt.imshow(normalized)
    i += 1

plt.show()
# -



# +
test_size = 0.3

train_set = example_images[0:floor(len(example_images) * (1 - test_size))]
test_set = example_images[-ceil(len(example_images) * test_size):]

# +
for path in train_set:
    shutil.copy(equalized_data_dir + path, recognition_train_images_dir)
    shutil.copy(os.path.basename(path)[:-5] + '.xml', recognition_train_annotations_dir)

for path in test_set:
    shutil.copy(equalized_data_dir + path, recognition_test_images_dir)
    shutil.copy(os.path.basename(path)[:-5] + '.xml', recognition_test_annotations_dir)
# -

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory = recognition_dir)
trainer.setTrainConfig(object_names_array=["ROI"], batch_size = 4, num_experiments = 10,
                       train_from_pretrained_model = pretrained_yolov3_model_path)
trainer.trainModel()

# +
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("/Users/zhuvikin/workspace/ai-university/data/chest_xray/recognition/models/detection_model-ex-08--loss-19.17.h5") 
detector.setJsonPath("/Users/zhuvikin/workspace/ai-university/data/chest_xray/recognition/json/detection_config.json")
detector.loadModel()
# -

base_data_dir + example_images[1]

detections = detector.detectObjectsFromImage(input_image=base_data_dir + example_images[6], 
                                             output_image_path="detected.jpg", 
                                             minimum_percentage_probability=50)
print(detections)

# +
# h_crop = 0.3
# v_crop = 0.3
# target_width = 256
# target_height = 256

# img = cv2.imread(file_names_normal[6], 0)

# plt.imshow(img)
# plt.title('Original')
# plt.show()

# print(img.shape)

# width = img.shape[0]
# height = img.shape[1]

# resized = cv2.resize(img, (int(target_width + 2 * target_width * h_crop),int(target_height + 2 * target_height * v_crop)))

# crop_h_from = floor(target_width * h_crop)
# crop_h_to = ceil(target_width + target_width * h_crop)

# crop_v_from = floor(target_height * v_crop)
# crop_v_to = ceil(target_height + target_height * v_crop)

# cropped = resized[crop_h_from:crop_h_to, crop_v_from:crop_v_to]

# plt.imshow(cropped)
# plt.title('Cropped')
# plt.show()
