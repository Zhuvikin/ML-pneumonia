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

i = 1
for example_image in example_images:
    ROI = ROIs[i - 1]
    img_path = equalized_data_dir + file_names_pneumonia[i - 1 + 0] # + 60
    img = cv2.imread(img_path, 0)
    width = img.shape[1]
    height = img.shape[0]
    nw = 512
    wc = width / nw
    nh = floor(height / wc)
    ch = floor(nh / 1)
    ct = floor((nh - ch) / 2)
    
    img2 = cv2.resize(img, (nw, nh), interpolation = cv2.INTER_AREA)
    img3 = img2[ct:ct + ch, :]
    img3 = cv2.equalizeHist(img3)
    
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
    
    x_start = min(white_columns)+10
    x_end = max(white_columns)-10
    img4g = img4[:,x_start:x_end]
    img4h = cv2.equalizeHist(img4g)
    
    img5 = adjust_gamma(img4h, 1.2)
    img5a = adjust_contrast(img5, 5)
    ret, img5b = cv2.threshold(img5a, 127, 255, cv2.THRESH_BINARY)
#     img5c = cv2.erode(img5b, kernel, iterations=3)
    
#     kernel = np.ones((5,5), np.uint8)
#     kernel[:,0:2] = 0
#     kernel[:,-2:] = 0
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    img5c = cv2.dilate(img5b, kernel2, iterations=1)
    img5d = cv2.morphologyEx(img5c, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
#     img5d = cv2.erode(img5d, kernel, iterations=4)
    
#     img5d = cv2.GaussianBlur(img5d, (3, 3), cv2.BORDER_REFLECT101)
#     img5d = cv2.resize(img5d, (nw, 5), interpolation = cv2.INTER_AREA)
#     img5d = cv2.resize(img5d, (nw, nh), interpolation = cv2.INTER_AREA)
#     ret, img5d = cv2.threshold(img5d, 70, 255, cv2.THRESH_BINARY)
    
#     img5b = cv2.Canny(img5a, 140, 200)
#     img5c = cv2.dilate(img5b, kernel, iterations=1)
    
#     kh = 2 * ceil(img5.shape[1] / 70) - 1    
#     img5d = cv2.Sobel(img5a, cv2.CV_64F, 1, 0, ksize = kh)
#     img5d = np.absolute(img5d)
#     img5d = np.uint8(255 * img5d / np.max(img5d))
#     img5d = cv2.equalizeHist(img5d)
    
    
    rows, cols = img5d.shape    
    non_empty_columns = np.where(img5d.max(axis=0) > 0)[0]
    x_from = min(non_empty_columns)
    x_to = max(non_empty_columns)
#     print(non_empty_columns)
#     print(x_from, x_to)
    img10 = img5a[:, x_from + 1:x_to]

    
    ax = plt.subplot(3, 6, i)
    plt.axis('off')
    plt.imshow(img5d)
    i += 1

plt.show()

# +
#     img5d = adjust_gamma(img5d, 0.7)
#     img5d = adjust_contrast(img5d, 10)
#     img5d = cv2.equalizeHist(img5d)
# #     ret, img5d = cv2.threshold(img5d, 120, 255, cv2.THRESH_BINARY)
#     img5d = cv2.erode(img5d, kernel, iterations=1)
# #     img5d = cv2.dilate(img5d, kernel, iterations=3)
#     img5d = cv2.GaussianBlur(img5d, (kh, kh), cv2.BORDER_REFLECT101)
#     ret, img5d = cv2.threshold(img5d, 80, 255, cv2.THRESH_BINARY)
#     img5d = cv2.erode(img5d, kernel, iterations=3)
#     img5d = cv2.dilate(img5d, kernel, iterations=3)
    
#     img5d = cv2.resize(img5d, (nw, 3), interpolation = cv2.INTER_LANCZOS4)


# #     img5d = cv2.GaussianBlur(img5d, (kh, kh), cv2.BORDER_REFLECT101)
# #     img5d = cv2.resize(img5d, (nw, 1), interpolation = cv2.INTER_LANCZOS4)
#     img5d = cv2.resize(img5d, (nw, nh), interpolation = cv2.INTER_AREA)
# #     ret, img5d = cv2.threshold(img5d, 130, 255, cv2.THRESH_BINARY)
    
# #     ret, img5d = cv2.threshold(img5d, 210, 255, cv2.THRESH_BINARY)
# #     img5d = cv2.GaussianBlur(img5d, (kh, kh), cv2.BORDER_REFLECT101)
# #     img5d = cv2.erode(img5d, kernel, iterations=3)
        
# #     img5d = cv2.resize(img5d, (nw, 5), interpolation = cv2.INTER_LANCZOS4)
# #     img5d = cv2.resize(img5d, (nw, 1), interpolation = cv2.INTER_AREA)
# #     img5d = cv2.resize(img5d, (nw, nh), interpolation = cv2.INTER_AREA)
# #     ret, img5d = cv2.threshold(img5d, 130, 255, cv2.THRESH_BINARY)
# #     img5d = cv2.erode(img5d, kernel, iterations=2)
    
# #     kernel = np.ones((5,5), np.uint8)
# #     kernel[:,0:2] = 0
# #     kernel[:,-2:] = 0
# #     print(kernel)
# #     img5c = cv2.dilate(img5b, kernel, iterations=2)
# #     img5d = cv2.GaussianBlur(img5c, (1, 103), cv2.BORDER_REFLECT101)
    
# #     img6 = cv2.GaussianBlur(img5b, (1, 1), cv2.BORDER_REFLECT101)
# #     img7 = cv2.resize(img6, (nw, 10), interpolation = cv2.INTER_AREA)

# #     img7 = cv2.resize(img5d, (nw, 3), interpolation = cv2.INTER_AREA)
# #     img7 = cv2.resize(img7, (nw, 1), interpolation = cv2.INTER_AREA)
# #     img8 = cv2.resize(img7, (nw, nh), interpolation = cv2.INTER_AREA)
# #     img8 = adjust_gamma(img8, 10)
    
# #     ret, img9 = cv2.threshold(img8, 150, 255, cv2.THRESH_BINARY)

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
