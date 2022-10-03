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
from matplotlib import pyplot as plt

cv2.__version__
# -

base_data_dir = '../data/chest_xray/'
data_dir = base_data_dir + 'prepeared/'

# +
img = cv2.imread(base_data_dir + 'train/NORMAL/IM-0115-0001.jpeg', 0)

plt.imshow(img)
plt.title('my picture')
plt.show()
# -

# Gather all images
file_names_normal = [path for path in glob.glob(base_data_dir + 'train/NORMAL/*.jpeg')] + [path for path in glob.glob('../data/chest_xray/test/NORMAL/*.jpeg')] + [path for path in glob.glob('../data/chest_xray/val/NORMAL/*.jpeg')]
file_names_pneumonia = [path for path in glob.glob(base_data_dir + 'train/PNEUMONIA/*.jpeg')] + [path for path in glob.glob('../data/chest_xray/test/PNEUMONIA/*.jpeg')] + [path for path in glob.glob('../data/chest_xray/val/PNEUMONIA/*.jpeg')]

image_sizes = [cv2.imread(image, 0).shape for image in file_names_normal + file_names_pneumonia]

# +
min_width = min([shape[0] for shape in image_sizes])
min_height = min([shape[1] for shape in image_sizes])
areas = [shape[0] * shape[1] for shape in image_sizes]

print('min width:', min_width)
print('min height:', min_height)

plt.figure(figsize=(16, 6))
plt.title('Distribution of Image Areas')
plt.hist(areas, bins = 100)
plt.show()
# -

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


