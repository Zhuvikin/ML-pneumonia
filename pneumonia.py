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

# # Pneumonia detection in chest X-Ray images

# ## Introduction
#
#

# The dataset consists of chest X-Rays given for patients with and without pneumonia. It is hard to detect disease by not educated person. Besides, it requires some time even for professional doctor to detect pneumonia. Thus, an automated approach would help with fast pneumonia diagnostic.
#
# There are two forms of pneumonia which are given in the dataset: bacterial and viral. The first task to solve is to find if patient has pneumonia at all (Binary Classification). Then the second task is to try to determine an actual form of the pneumonia if any (Multi-Class Classification.

# ## Dataset overview

# +
import os
import random
import numpy as np
import tensorflow as tf
from keras import backend as K
import cv2
import glob
import pandas as pd
from matplotlib import pyplot as plt
from math import ceil, floor
from tqdm import tqdm, tqdm_notebook
from skimage import morphology
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from imblearn.under_sampling import RandomUnderSampler
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import label_binarize

from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D, ZeroPadding2D, Dense, Dropout, \
    Flatten, Input, LSTM, TimeDistributed
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import recall_score, auc, roc_auc_score, f1_score, roc_curve, classification_report, \
    confusion_matrix

# -

# We define global constants where the needed directories paths are stored. Then we create the directories if some are not exist

# +
base_data_dir = '../data/chest_xray/'

data_dir = base_data_dir + 'prepared/'
normal_dir = data_dir + 'normal/'
pneumonia_dir = data_dir + 'pneumonia/'
bacteria_dir = pneumonia_dir + 'bacteria/'
virus_dir = pneumonia_dir + 'virus/'
models_dir = './models/'
models_lungs_dir = models_dir + 'lung/'
models_binary_dir = models_dir + 'binary/'
models_categorical_dir = models_dir + 'categorical/'

for directory in [normal_dir, bacteria_dir, virus_dir, models_lungs_dir,
                  models_binary_dir, models_categorical_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

lung_segmentation_model_path = models_lungs_dir + 'lung_segmentation.hdf5'
# -

# The initial dataset already have some separation of the test, train and validation sets. We gather images from all these directories because we will use our own split.

# +
normal_patterns = [
    base_data_dir + 'train/NORMAL/*.jpeg',
    base_data_dir + 'test/NORMAL/*.jpeg',
    base_data_dir + 'val/NORMAL/*.jpeg'
]

bacteria_patterns = [
    base_data_dir + 'train/PNEUMONIA/*bacteria*.jpeg',
    base_data_dir + 'test/PNEUMONIA/*bacteria*.jpeg',
    base_data_dir + 'val/PNEUMONIA/*bacteria*.jpeg'
]

virus_patterns = [
    base_data_dir + 'train/PNEUMONIA/*virus*.jpeg',
    base_data_dir + 'test/PNEUMONIA/*virus*.jpeg',
    base_data_dir + 'val/PNEUMONIA/*virus*.jpeg'
]

raw_normal = [item for sublist in [glob.glob(path) for path in normal_patterns] for item in sublist]
raw_bacteria = [item for sublist in [glob.glob(path) for path in bacteria_patterns] for item in sublist]
raw_virus = [item for sublist in [glob.glob(path) for path in virus_patterns] for item in sublist]
# -

# Plot a few examples of the normal images and images with bacterial and viral pneumonias 

# +
preview_n = 8
preview_n_i = [0, 10, 20, 30]
preview_b_i = [0, 10]
preview_v_i = [0, 10]

preview_paths = np.array(raw_normal)[preview_n_i].tolist() + np.array(raw_bacteria)[preview_b_i].tolist() + \
                np.array(raw_virus)[preview_v_i].tolist()

fig = plt.figure(figsize = (17, 8))
plt.suptitle("Raw X-Ray Images", size = 22)
for i in range(2):
    for k in range(preview_n // 2):
        index = i * preview_n // 2 + k
        img = cv2.imread(preview_paths[index], 0)
        ax = plt.subplot(2, preview_n // 2, index + 1)
        im = plt.imshow(img, cmap = 'bone')
        if index < preview_n // 2:
            plt.title('Normal')
        elif index < preview_n // 2 + preview_n // 4:
            plt.title('Bacterial')
        else:
            plt.title('Viral')

plt.subplots_adjust(hspace = 0.3)
plt.show()
# -

# As far as we can see that dataset consists of images with various image sizes, some images are not symmetric and contain a lot of useless information around except for lungs.
#
# We need to preprocess images first in order to get rid of the useless information and to normalize images. The lung segmentation algorithm available at https://github.com/imlab-uiip/lung-segmentation-2d is used for this purpose. This is already trained UNet neural network which is for the specific purpose of lung segmentation.

# +
get_file('lung_segmentation.hdf5',
         'https://raw.githubusercontent.com/imlab-uiip/lung-segmentation-2d/master/trained_model.hdf5',
         cache_dir = models_dir, cache_subdir = 'lung')

UNet = load_model(lung_segmentation_model_path)


# -

# However, the used lung segmentation model does not recognize the correct lungs areas all the time and we use some lungs mask post-processing in order to make sure we do not crop usefull information in the images

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
    img = adjust_gamma(img, 0.75)
    img = img * 1.2
    img = np.clip(img, 0, 255)

    img2 = img.astype(float)
    img2 -= img2.mean()
    img2 /= img2.std()

    img2 = np.expand_dims(img2, -1)
    inp_shape = img2.shape
    pred = UNet.predict([[img2]])[..., 0].reshape(inp_shape[:2])
    pr = pred > 0.5
    pr = remove_small_regions(pr, 0.01 * np.prod(im_shape))
    pr = pr.astype(int)

    non_empty_columns = np.where(pr.max(axis = 0) > 0)[0]
    non_empty_rows = np.where(pr.max(axis = 1) > 0)[0]
    left = min(non_empty_columns) if len(non_empty_columns) > 0 else 0
    right = max(non_empty_columns) if len(non_empty_columns) > 0 else 256
    top = min(non_empty_rows) if len(non_empty_rows) > 0 else 0
    bottom = max(non_empty_rows) if len(non_empty_rows) > 0 else 256

    if right < 256 / 1.9:
        right = 256 - left

    if left > 256 / 2.1:
        left = 256 - right

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

# Normalize dataset and save in the `prepeared` folder

# +
image_width = 256

i = 0
for path in tqdm_notebook(raw_normal):
    output_path = normal_dir + '{0:04d}.jpeg'.format(i)
    if not os.path.exists(output_path):
        img = cv2.imread(path, 0)
        normalized = normalize_image(img, image_width)
        cv2.imwrite(output_path, normalized)
    i += 1

i = 0
for path in tqdm_notebook(raw_bacteria + raw_virus):
    if 'bacteria' in path:
        output_path = bacteria_dir + '{0:04d}.jpeg'.format(i)
    else:
        output_path = virus_dir + '{0:04d}.jpeg'.format(i)
    if not os.path.exists(output_path):
        img = cv2.imread(path, 0)
        normalized = normalize_image(img, image_width)
        cv2.imwrite(output_path, normalized)
    i += 1
# + {}
normal_df = pd.DataFrame(
    {'path': glob.glob(normal_dir + '*.jpeg'), 'normal': 1, 'bacteria': 0, 'virus': 0, 'target': 'Normal'})
bacteria_df = pd.DataFrame(
    {'path': glob.glob(bacteria_dir + '*.jpeg'), 'normal': 0, 'bacteria': 1, 'virus': 0, 'target': 'Pneumonia'})
virus_df = pd.DataFrame(
    {'path': glob.glob(virus_dir + '*.jpeg'), 'normal': 0, 'bacteria': 0, 'virus': 1, 'target': 'Pneumonia'})

# normal_df = pd.DataFrame(
#     {'path': raw_normal, 'normal': 1, 'bacteria': 0, 'virus': 0, 'target': 'Normal'})
# bacteria_df = pd.DataFrame(
#     {'path': raw_bacteria, 'normal': 0, 'bacteria': 1, 'virus': 0, 'target': 'Pneumonia'})
# virus_df = pd.DataFrame(
#     {'path': raw_virus, 'normal': 0, 'bacteria': 0, 'virus': 1, 'target': 'Pneumonia'})
# -


# Compare the previously shown raw images with the normalized versions

# +
processed_previews = np.array(list(normal_df.sort_values('path').path))[preview_n_i].tolist() + \
                     np.array(list(bacteria_df.sort_values('path').path))[preview_b_i].tolist() + \
                     np.array(list(virus_df.sort_values('path').path))[preview_v_i].tolist()
fig = plt.figure(figsize = (17, 8))
plt.suptitle("Normalized X-Ray Images", size = 22)
for i in range(2):
    for k in range(preview_n // 2):
        index = i * preview_n // 2 + k
        img = cv2.imread(processed_previews[index], 0)
        ax = plt.subplot(2, preview_n // 2, index + 1)
        im = plt.imshow(img, cmap = 'bone')
        if index < preview_n // 2:
            plt.title('Normal')
        elif index < preview_n // 2 + preview_n // 4:
            plt.title('Bacterial')
        else:
            plt.title('Viral')

plt.subplots_adjust(hspace = 0.3)
plt.show()
# -

dataset = pd.concat([normal_df, bacteria_df, virus_df])
dataset = dataset.sort_values('path')
dataset = dataset.sample(frac = 1, random_state = 0).reset_index(drop = True)

pd.set_option('max_colwidth', 100)
dataset.head(10)

# Let us check if there are common features of normal images and images with pneumonia. We perform Karhunen Loeve Decomposition for the first `1500` images for both classes and plot first few principal component images

# +
number_of_images = 1500
n_row, n_col = 1, 6
n_components = n_row * n_col

pca = PCA(n_components = n_components, svd_solver = 'randomized', whiten = True)

plt.rcParams.update({'font.size': 16})


def plot_gallery(title, images, n_col = n_col, n_row = n_row, cmap = plt.cm.gray):
    plt.figure(figsize = (2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size = 16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(256, 256), cmap = 'bone',
                   interpolation = 'nearest',
                   vmin = -vmax, vmax = vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.02, -0.1, 0.99, 0.93, 0.04, 0.)


few_normal_images = np.array([cv2.imread(path, 0).flatten() for path in normal_df.head(number_of_images).path])
pca.fit(few_normal_images)
plot_gallery('First few principal components of normal lungs', pca.components_[:n_components])
plt.show()

print('One can even clearly observe bronchus system in few most significant components of normal lungs')

pneumonia_df = pd.concat([bacteria_df, virus_df]).sample(frac = 1, random_state = 0).reset_index(drop = True)
few_pneumonia_images = np.array([cv2.imread(path, 0).flatten() for path in pneumonia_df.head(number_of_images).path])
pca.fit(few_pneumonia_images)
plot_gallery('First few principal components of lungs with pneumonia', pca.components_[:n_components])
plt.show()

print(
    'Pleural effusions and airspace consolidation in the different parts of the lungs with pneumonia make few first principal components a bit blurred')
# -

# Check the balance of the classes in the dataset

# +
colors = ['#D4FCEF', '#FDAFD7', '#FECBE5']


def labels(normal_amount, bacteria_amount, virus_amount): return \
    'Normal ({0:d})'.format(normal_amount), \
    'Bacteria ({0:d})'.format(bacteria_amount), \
    'Virus ({0:d})'.format(virus_amount)


def sizes(normal_amount, bacteria_amount, virus_amount):
    total = normal_amount + bacteria_amount + virus_amount
    return [normal_amount / total, bacteria_amount / total, virus_amount / total]


fig1, ax1 = plt.subplots()
ax1.pie(sizes(normal_df.shape[0], bacteria_df.shape[0], virus_df.shape[0]),
        explode = (0.1, 0, 0), labels = labels(normal_df.shape[0], bacteria_df.shape[0], virus_df.shape[0]),
        autopct = '%1.1f%%', startangle = 90, colors = colors)
ax1.axis('equal')
ax1.title.set_text('Data Set')
plt.show()
# -

# ## 1. Normal vs Pneumonia Binary Classification

# In order to perform binary classification it is better to train model with balanced data. Undersampling and split to the train, test and validation dataset are performed as follows

# +
colors = ['#D4FCEF', '#FDAFD7', '#FECBE5']

test_size = 0.2
validation_size = 0.02

sampler = RandomUnderSampler(random_state = 0)
X_balanced, _ = sampler.fit_resample(dataset[['path']].values, dataset[['target']].values)

balanced_dataset = pd.DataFrame({'path': X_balanced[:, 0]})
balanced_dataset = balanced_dataset.sample(frac = 1, random_state = 0).reset_index(drop = True)
balanced_dataset = pd.merge(balanced_dataset, dataset, on = 'path')
rest_dataset = pd.DataFrame(
    {'path': list(set(X_balanced[:, 0].tolist()).symmetric_difference(dataset[['path']].values[:, 0].tolist()))})
rest_dataset = rest_dataset.sample(frac = 1, random_state = 0).reset_index(drop = True)
rest_dataset = pd.merge(rest_dataset, dataset, on = 'path')

X_train_validation, X_test, y_train_validation, _ = train_test_split(
    balanced_dataset.path.values, balanced_dataset.target.values,
    test_size = test_size, random_state = 0)
X_train, X_validation, _, _ = train_test_split(
    X_train_validation, y_train_validation,
    test_size = validation_size, random_state = 3)

train_dataset = pd.merge(pd.DataFrame({'path': X_train}), dataset, on = 'path')
validation_dataset = pd.merge(pd.DataFrame({'path': X_validation}), dataset, on = 'path')
test_dataset = pd.concat([pd.merge(pd.DataFrame({'path': X_test}), dataset, on = 'path'), rest_dataset])

fig = plt.figure(figsize = (17, 4))
ax1 = fig.add_subplot(131)
ax1.pie(sizes(train_dataset.sum()[1], train_dataset.sum()[2], train_dataset.sum()[3]),
        explode = (0.1, 0, 0), labels = labels(train_dataset.sum()[1], train_dataset.sum()[2], train_dataset.sum()[3]),
        autopct = '%1.1f%%', startangle = 90, colors = colors)
ax1.axis('equal')
ax1.title.set_text('Train Set')

ax3 = fig.add_subplot(132)
ax3.pie(sizes(validation_dataset.sum()[1], validation_dataset.sum()[2], validation_dataset.sum()[3]),
        explode = (0.1, 0, 0),
        labels = labels(validation_dataset.sum()[1], validation_dataset.sum()[2], validation_dataset.sum()[3]),
        autopct = '%1.1f%%', startangle = 90, colors = colors)
ax3.axis('equal')
ax3.title.set_text('Validation Set')

ax2 = fig.add_subplot(133)
ax2.pie(sizes(test_dataset.sum()[1], test_dataset.sum()[2], test_dataset.sum()[3]),
        explode = (0.1, 0, 0), labels = labels(test_dataset.sum()[1], test_dataset.sum()[2], test_dataset.sum()[3]),
        autopct = '%1.1f%%', startangle = 90, colors = colors)
ax2.axis('equal')
ax2.title.set_text('Test Set')

plt.subplots_adjust(wspace = 1)
plt.show()
# -

# In order to increase an amount of the train data we use image generators with horizontal flip augmentation

# +
imageGenerator = ImageDataGenerator(rescale = 1. / 255, horizontal_flip = True)
testGenerator = ImageDataGenerator(rescale = 1. / 255)

batch_size = 4
x_col = 'path'
y_col = 'target'
classes = ['Normal', 'Pneumonia']
mode = 'grayscale'
target_size = (150, 150)

print('Train generator:')
train_generator = imageGenerator.flow_from_dataframe(train_dataset, x_col = x_col, y_col = y_col,
                                                     seed = 0, target_size = target_size, batch_size = batch_size,
                                                     class_mode = 'binary', color_mode = mode)

print('\nValidation generator:')
validation_generator = testGenerator.flow_from_dataframe(validation_dataset, x_col = x_col, y_col = y_col,
                                                         seed = 0, target_size = target_size, batch_size = batch_size,
                                                         class_mode = 'binary', color_mode = mode, shuffle = False)

print('\nTest generator:')
test_generator = testGenerator.flow_from_dataframe(test_dataset, x_col = x_col, y_col = y_col,
                                                   seed = 0, target_size = target_size, batch_size = 1,
                                                   class_mode = 'binary', color_mode = mode, shuffle = False)

# +
model = Sequential()

model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = train_generator.image_shape))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(rate = 0.15))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(rate = 0.15))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(rate = 0.15))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(rate = 0.15))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.15))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.15))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()


# -

# Define custom metrics

# +
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


model_check_point = ModelCheckpoint(
    models_binary_dir + 'pneumonia-{val_loss:.2f}-{val_acc:.2f}-{val_precision:.2f}-{val_recall:.2f}.hdf5',
    save_best_only = True, verbose = 1, monitor = 'val_acc', mode = 'max')
# -

# Train neural network

# +
optimizer = Adam()

model.compile(loss = 'binary_crossentropy', optimizer = optimizer,
              metrics = ['accuracy', recall, precision])

train_generator.reset()
validation_generator.reset()
history = model.fit_generator(epochs = 100, shuffle = True, validation_data = validation_generator,
                              steps_per_epoch = 100, generator = train_generator,
                              validation_steps = validation_dataset.shape[0] * batch_size,
                              verbose = 1, callbacks = [model_check_point])
# -

# Plot the dependencies of loss, accuracy, recall and precision on the training epoch number

# +
metrics = ['loss', 'acc', 'recall', 'precision']

fig = plt.figure(figsize = (17, 17))
for i, metric in enumerate(metrics):
    ax = plt.subplot(2, 2, i + 1)
    ax.set_facecolor('w')
    ax.grid(b = False)
    ax.plot(history.history[metric], color = '#00bf81')
    ax.plot(history.history['val_' + metric], color = '#ff0083')
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')

plt.show()

# +
binary_models = []
for model_path in glob.glob(models_binary_dir + 'pneumonia-*.hdf5'):
    name = os.path.basename(model_path)
    (prefix, sep, suffix) = name.rpartition('.')
    scores = list(map(lambda k: float(k), prefix.split('-')[1:]))
    binary_models.append([model_path] + scores)

binary_models = pd.DataFrame(binary_models, columns = ['path', 'loss', 'acc', 'precision', 'recall'])

# sorted_models = binary_models.sort_values(['acc', 'loss'], ascending = [False, True])
sorted_models = binary_models.sort_values(['acc', 'precision'], ascending = [False, False])

best_binary_model_path = sorted_models.path.iloc[0]
print('Best binary model:', best_binary_model_path)
# -

best_binary_model = load_model(best_binary_model_path, custom_objects = {
    'recall': recall,
    'precision': precision
})

test_generator.reset()
test_pred = best_binary_model.predict_generator(test_generator, verbose = 1, steps = test_dataset.shape[0])

print(classification_report(test_generator.classes, np.rint(test_pred).astype(int).flatten(), target_names = classes))

# +
cm = confusion_matrix(y_true = test_generator.classes, y_pred = np.rint(test_pred).astype(int).flatten().tolist())
ncm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

fig = plt.figure(figsize = (17, 7))

ax = plt.subplot(1, 2, 1)
sns.heatmap(cm, annot = True, annot_kws = {"size": 20}, ax = ax, fmt = 'd',
            cmap = sns.dark_palette((30 / 256, 234 / 256, 186 / 256), input = "rgb"))

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(classes)
ax.yaxis.set_ticklabels(classes)

ax = plt.subplot(1, 2, 2)
sns.heatmap(ncm, annot = True, annot_kws = {"size": 20}, ax = ax, fmt = 'f',
            cmap = sns.dark_palette((30 / 256, 234 / 256, 186 / 256), input = "rgb"))

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Normalized Confusion Matrix')
ax.xaxis.set_ticklabels(classes)
ax.yaxis.set_ticklabels(classes)

plt.show()

# +
fpr, tpr, thresholds = roc_curve(test_generator.classes, test_pred)

fig = plt.figure(figsize = (10, 10))
lw = 2
plt.plot(fpr, tpr, color = '#ff0083',
         lw = lw, label = 'ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color = 'black', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.005])
plt.ylim([0.0, 1.005])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc = "lower right")
plt.show()
# -

# ## 2. Normal vs Bacterial vs Viral Classification

# In order to find multi-class classification model let us rebalance dataset again to have the same portions for normal, bacterial and viral classes 

# +
test_size = 0.2
validation_size = 0.02

multi_class_dataset = dataset.copy()
multi_class_dataset['target'] = multi_class_dataset.apply(
    lambda row: 'Normal' if row['normal'] == 1 else 'Bacterial' if row['bacteria'] == 1 else 'Viral', axis = 1)

sampler = RandomUnderSampler(random_state = 0)
X_balanced, _ = sampler.fit_resample(multi_class_dataset[['path']].values, multi_class_dataset[['target']].values)

balanced_multi_class_dataset = pd.DataFrame({'path': X_balanced[:, 0]})
balanced_multi_class_dataset = balanced_multi_class_dataset.sample(frac = 1, random_state = 0).reset_index(drop = True)
balanced_multi_class_dataset = pd.merge(balanced_multi_class_dataset, multi_class_dataset, on = 'path')

rest_multi_class_dataset = pd.DataFrame(
    {'path': list(
        set(X_balanced[:, 0].tolist()).symmetric_difference(multi_class_dataset[['path']].values[:, 0].tolist()))})
rest_multi_class_dataset = rest_multi_class_dataset.sample(frac = 1, random_state = 0).reset_index(drop = True)
rest_multi_class_dataset = pd.merge(rest_multi_class_dataset, multi_class_dataset, on = 'path')

X_train_validation, X_test, y_train_validation, _ = train_test_split(
    balanced_multi_class_dataset.path.values, balanced_multi_class_dataset.target.values,
    test_size = test_size, random_state = 3)

X_train, X_validation, _, _ = train_test_split(
    X_train_validation, y_train_validation,
    test_size = validation_size, random_state = 2)

train_multi_class_dataset = pd.merge(pd.DataFrame({'path': X_train}), multi_class_dataset, on = 'path')
validation_multi_class_dataset = pd.merge(pd.DataFrame({'path': X_validation}), multi_class_dataset, on = 'path')
test_multi_class_dataset = pd.concat(
    [pd.merge(pd.DataFrame({'path': X_test}), multi_class_dataset, on = 'path'), rest_multi_class_dataset])

fig = plt.figure(figsize = (17, 4))
ax1 = fig.add_subplot(131)
ax1.pie(
    sizes(train_multi_class_dataset.sum()[1], train_multi_class_dataset.sum()[2], train_multi_class_dataset.sum()[3]),
    explode = (0.1, 0, 0), labels = labels(train_multi_class_dataset.sum()[1], train_multi_class_dataset.sum()[2],
                                           train_multi_class_dataset.sum()[3]),
    autopct = '%1.1f%%', startangle = 90, colors = colors)
ax1.axis('equal')
ax1.title.set_text('Train Set')

ax3 = fig.add_subplot(132)
ax3.pie(sizes(validation_multi_class_dataset.sum()[1], validation_multi_class_dataset.sum()[2],
              validation_multi_class_dataset.sum()[3]),
        explode = (0.1, 0, 0),
        labels = labels(validation_multi_class_dataset.sum()[1], validation_multi_class_dataset.sum()[2],
                        validation_multi_class_dataset.sum()[3]),
        autopct = '%1.1f%%', startangle = 90, colors = colors)
ax3.axis('equal')
ax3.title.set_text('Validation Set')

ax2 = fig.add_subplot(133)
ax2.pie(sizes(test_multi_class_dataset.sum()[1], test_multi_class_dataset.sum()[2], test_multi_class_dataset.sum()[3]),
        explode = (0.1, 0, 0), labels = labels(test_multi_class_dataset.sum()[1], test_multi_class_dataset.sum()[2],
                                               test_multi_class_dataset.sum()[3]),
        autopct = '%1.1f%%', startangle = 90, colors = colors)
ax2.axis('equal')
ax2.title.set_text('Test Set')

plt.subplots_adjust(wspace = 1)
plt.show()

# +
imageGenerator = ImageDataGenerator(rescale = 1. / 255, horizontal_flip = True)
testGenerator = ImageDataGenerator(rescale = 1. / 255)

batch_size = 4
x_col = 'path'
y_col = 'target'
multi_classes = ['Normal', 'Bacterial', 'Viral']
mode = 'grayscale'
target_size = (150, 150)

print('Train generator:')
train_multi_class_generator = imageGenerator.flow_from_dataframe(train_multi_class_dataset, x_col = x_col,
                                                                 y_col = y_col, classes = multi_classes,
                                                                 seed = 0, target_size = target_size,
                                                                 batch_size = batch_size,
                                                                 class_mode = 'categorical', color_mode = mode)

print('\nValidation generator:')
validation_multi_class_generator = testGenerator.flow_from_dataframe(validation_multi_class_dataset, x_col = x_col,
                                                                     y_col = y_col, classes = multi_classes,
                                                                     seed = 0, target_size = target_size,
                                                                     batch_size = batch_size,
                                                                     class_mode = 'categorical', color_mode = mode,
                                                                     shuffle = False)

print('\nTest generator:')
test_multi_class_generator = testGenerator.flow_from_dataframe(test_multi_class_dataset, x_col = x_col, y_col = y_col,
                                                               seed = 0, target_size = target_size, batch_size = 1,
                                                               class_mode = 'categorical', color_mode = mode,
                                                               shuffle = False, classes = multi_classes)
# -

# We will keep network architecture the same but change the last layer to have 3 neurons only 

# +
multi_class_model = Sequential()

multi_class_model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = train_generator.image_shape))
multi_class_model.add(MaxPooling2D((2, 2)))
multi_class_model.add(BatchNormalization())
multi_class_model.add(Dropout(rate = 0.15))

multi_class_model.add(Conv2D(64, (3, 3), activation = 'relu'))
multi_class_model.add(MaxPooling2D((2, 2)))
multi_class_model.add(BatchNormalization())
multi_class_model.add(Dropout(rate = 0.15))

multi_class_model.add(Conv2D(128, (3, 3), activation = 'relu'))
multi_class_model.add(MaxPooling2D((2, 2)))
multi_class_model.add(BatchNormalization())
multi_class_model.add(Dropout(rate = 0.15))

multi_class_model.add(Conv2D(128, (3, 3), activation = 'relu'))
multi_class_model.add(MaxPooling2D((2, 2)))
multi_class_model.add(BatchNormalization())
multi_class_model.add(Dropout(rate = 0.15))

multi_class_model.add(Conv2D(128, (3, 3), activation = 'relu'))
multi_class_model.add(MaxPooling2D((2, 2)))

multi_class_model.add(GlobalAveragePooling2D())
multi_class_model.add(Dense(64, activation = 'relu'))
multi_class_model.add(Dropout(0.15))

multi_class_model.add(Dense(128, activation = 'relu'))
multi_class_model.add(Dropout(0.15))
multi_class_model.add(Dense(len(multi_classes), activation = 'softmax'))

multi_class_model.summary()
# -

multi_class_model_check_point = ModelCheckpoint(
    models_categorical_dir + 'pneumonia-{val_loss:.2f}-{val_acc:.2f}-{val_precision:.2f}-{val_recall:.2f}.hdf5',
    save_best_only = True, verbose = 1, monitor = 'val_acc', mode = 'max')

# +
multi_class_optimizer = Adam()

multi_class_model.compile(loss = 'categorical_crossentropy', optimizer = multi_class_optimizer,
                          metrics = ['accuracy', recall, precision])

train_multi_class_generator.reset()
validation_multi_class_generator.reset()
multi_class_history = multi_class_model.fit_generator(epochs = 100, shuffle = True,
                                                      validation_data = validation_multi_class_generator,
                                                      steps_per_epoch = 100,
                                                      generator = train_multi_class_generator,
                                                      validation_steps = validation_multi_class_dataset.shape[
                                                                             0] * batch_size,
                                                      verbose = 1, callbacks = [multi_class_model_check_point])

# +
metrics = ['loss', 'acc', 'recall', 'precision']

fig = plt.figure(figsize = (17, 17))
for i, metric in enumerate(metrics):
    ax = plt.subplot(2, 2, i + 1)
    ax.set_facecolor('w')
    ax.grid(b = False)
    ax.plot(multi_class_history.history[metric], color = '#00bf81')
    ax.plot(multi_class_history.history['val_' + metric], color = '#ff0083')
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')

plt.show()

# +
categorical_models = []
for model_path in glob.glob(models_categorical_dir + 'pneumonia-*.hdf5'):
    name = os.path.basename(model_path)
    (prefix, sep, suffix) = name.rpartition('.')
    scores = list(map(lambda k: float(k), prefix.split('-')[1:]))
    categorical_models.append([model_path] + scores)

categorical_models = pd.DataFrame(categorical_models, columns = ['path', 'loss', 'acc', 'precision', 'recall'])

# sorted_models = categorical_models.sort_values(['acc', 'loss'], ascending = [False, True])
sorted_models = categorical_models.sort_values(['acc', 'precision'], ascending = [False, False])

best_categorical_model_path = sorted_models.path.iloc[0]
print('Best categorical model:', best_categorical_model_path)
# -

best_categorical_model = load_model(best_categorical_model_path, custom_objects = {
    'recall': recall,
    'precision': precision
})

test_multi_class_generator.reset()
test_multi_class_pred = best_categorical_model.predict_generator(test_multi_class_generator, verbose = 1,
                                                                 steps = test_multi_class_dataset.shape[0])

class_labels = ['Bacterial', 'Normal', 'Viral']
print(classification_report(test_multi_class_generator.classes, np.argmax(test_multi_class_pred, axis = 1),
                            target_names = class_labels))

# +
cm_multi_class = confusion_matrix(y_true = test_multi_class_generator.classes,
                                  y_pred = np.argmax(test_multi_class_pred, axis = 1))
ncm_multi_class = cm_multi_class.astype('float') / cm_multi_class.sum(axis = 1)[:, np.newaxis]

fig = plt.figure(figsize = (17, 7))

ax = plt.subplot(1, 2, 1)
sns.heatmap(cm_multi_class, annot = True, annot_kws = {"size": 20}, ax = ax, fmt = 'd',
            cmap = sns.dark_palette((30 / 256, 234 / 256, 186 / 256), input = "rgb"))

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(class_labels)
ax.yaxis.set_ticklabels(class_labels)

ax = plt.subplot(1, 2, 2)
sns.heatmap(ncm_multi_class, annot = True, annot_kws = {"size": 20}, ax = ax, fmt = 'f',
            cmap = sns.dark_palette((30 / 256, 234 / 256, 186 / 256), input = "rgb"))

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Normalized Confusion Matrix')
ax.xaxis.set_ticklabels(class_labels)
ax.yaxis.set_ticklabels(class_labels)

plt.show()

# +
mc_pred = test_multi_class_pred
mc_true = label_binarize(test_multi_class_generator.classes, classes = [0, 1, 2])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(multi_classes)):
    fpr[i], tpr[i], _ = roc_curve(mc_true[:, i], mc_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fig = plt.figure(figsize = (10, 10))
lw = 2

colors = ['#ff0083', '#00d791', '#ff7ec0']

for i in range(len(multi_classes)):
    plt.plot(fpr[i], tpr[i], color = colors[i],
             lw = lw, label = class_labels[i] + ' - ROC curve (area = %0.2f)' % roc_auc[i])

plt.plot([0, 1], [0, 1], color = 'black', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.005])
plt.ylim([0.0, 1.005])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc = "lower right")
plt.show()
# -

# ## Conclusion
#
# We studied the two approaches of chest X-ray images classification. The former tries to find the fact of disease only which belongs to the binary classification problems class. In addidion, the latter tries to distinguish between the two forms of pneumonia: bacterial and viral. While the fact of disease is concluded with acceptable probabiliy, the specific form is the more difficult task for the considered network. An algorithm has more mistakes while deal with viral form.
