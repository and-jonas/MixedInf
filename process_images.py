
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 15.03.2021
# Extract training data from labelled images and train segmentation algorithm
# ======================================================================================================================

import cv2
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import os
import glob
import imageio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
from sklearn.model_selection import GridSearchCV  # Create the parameter grid based on the results of random search
import SegmentationFunctions
import utils
from importlib import reload
reload(utils)
reload(SegmentationFunctions)

# ======================================================================================================================
# (0) pre-process images (not used)
# ======================================================================================================================

# get image names
workdir = "P:/Public/Jonas/004_Divers/Test2"
image_paths = glob.glob(f'{workdir}/Evaluation/orig/*.jpg')
image_names = []
for p in image_paths:
    image_name = os.path.splitext(os.path.basename(p))[0]
    image_names.append(image_name)

for image_name in image_names:
    # mask = imageio.imread(f'{workdir}/Processed/Leaf_masks/{image_name}.tiff')
    image = imageio.imread(f'{workdir}/Evaluation/orig/{image_name}.jpg')
    image_norm = utils.preprocess_image(image)
    image_norm_norm = utils.normalize_image_illumination(image_norm)
    imageio.imwrite(f'{workdir}/Evaluation/orig_pp/{image_name}.tiff', image_norm_norm)

# ======================================================================================================================
# (0) get training patches from pp images
# ======================================================================================================================

dir = "Z:/Public/Jonas/004_Divers/Test2/Green/Control"
dir_BGR = "Z:/Public/Jonas/004_Divers/Test2/Training_images/BGR2"

files = [os.path.basename(file) for file in glob.glob(f'{dir}/*.png')]

for file in files:
    # get image
    path = glob.glob(f'{dir_BGR}/{file.replace(".png", "")}*.jpg')[0]
    img = imageio.imread(path)
    # get roi coordinates
    path = glob.glob(f'{dir}/{file.replace(".png", "")}*.csv')[0]
    rois = pd.read_csv(path)
    i = 0
    for row in rois.iterrows():
        print(i)
        r = row[1]
        new_img = img[r[1]:r[3], r[0]:r[2]]
        if r[4] == 'negative':
            sink_dir = "Z:/Public/Jonas/004_Divers/Test2/Green/Neg/BGR"
        elif r[4] == "positive":
            sink_dir = "Z:/Public/Jonas/004_Divers/Test2/Green/Pos/BGR"
        path = f'{sink_dir}/{file.replace(".png", "")}_{i+1}bgr.png'
        imageio.imsave(path, new_img)
        i += 1

# ======================================================================================================================
# (1) extract color features and save to .csv
# ======================================================================================================================

workdir = 'Z:/Public/Jonas/004_Divers/Test2/Green'

# set directories to previously selected training patches
dir_positives = f'{workdir}/Pos'
dir_negatives = f'{workdir}/Neg'

# extract feature data for all pixels in all patches
# output is stores in .csv files in the same directories
SegmentationFunctions.iterate_patches(dir_positives, dir_negatives)

# set directories to previously selected training patches
dir_positives = f'{workdir}/Pos/BGR'
dir_negatives = f'{workdir}/Neg/BGR'
SegmentationFunctions.iterate_patches(dir_positives, dir_negatives)

# ======================================================================================================================
# (2) combine training data from all patches into single file
# ======================================================================================================================

# set directories to previously selected training patches
dir_positives = f'{workdir}/Pos'
dir_negatives = f'{workdir}/Neg'

# import all training data
# get list of files
files_pos = glob.glob(f'{dir_positives}/*.csv')
files_neg = glob.glob(f'{dir_negatives}/*.csv')

# load data
train_data = []
for i, file in enumerate(files_pos):
    print(i)
    data = pd.read_csv(file)
    # data = data.iloc[::10, :]  # only keep every 10th pixel of the patch
    train_data.append(data)
# to single df
train_data_pos_orig = pd.concat(train_data)

# load data
train_data = []
for i, file in enumerate(files_neg):
    print(i)
    data = pd.read_csv(file)
    data = data.iloc[::4, :]  # only keep every 10th pixel of the patch
    train_data.append(data)
# to single df
train_data_neg_orig = pd.concat(train_data)

# set directories to previously selected training patches
dir_positives = f'{workdir}/Pos/BGR'
dir_negatives = f'{workdir}/Neg/BGR'

files_pos = glob.glob(f'{dir_positives}/*.csv')
files_neg = glob.glob(f'{dir_negatives}/*.csv')

# load data
train_data = []
for i, file in enumerate(files_pos):
    print(i)
    data = pd.read_csv(file)
    # data = data.iloc[::10, :]  # only keep every 10th pixel of the patch
    train_data.append(data)
# to single df
train_data_pos_pp = pd.concat(train_data)

# load data
train_data = []
for i, file in enumerate(files_neg):
    print(i)
    data = pd.read_csv(file)
    data = data.iloc[::4, :]  # only keep every 10th pixel of the patch
    train_data.append(data)
# to single df
train_data_neg_pp = pd.concat(train_data)

train_data_orig = pd.concat([train_data_pos_orig, train_data_neg_orig]).add_suffix('_orig')
train_data_pp = pd.concat([train_data_pos_pp, train_data_neg_pp]).add_suffix('_pp')

train_data = pd.concat([train_data_orig.reset_index(drop=True), train_data_pp.reset_index(drop=True)], axis=1)
train_data = train_data.drop(['response_orig'], axis=1)
train_data = train_data.rename(columns={'response_pp': 'response'})

# export, this may take a while
train_data.to_csv(f'{workdir}/training_data.csv', index=False)

# ======================================================================================================================
# (3) train random forest classifier
# ======================================================================================================================

train_data = pd.read_csv(f'{workdir}/training_data.csv')

# OPTIONAL: sample an equal number of rows per class for training
n_pos = train_data.groupby('response').count().iloc[0, 0]
n_neg = train_data.groupby('response').count().iloc[1, 0]
n_min = min(n_pos, n_neg)
train_data = train_data.groupby(['response']).apply(lambda grp: grp.sample(n=n_min))

# n_estimators = [int(x) for x in np.linspace(start=20, stop=200, num=10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4, 8]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]  # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
#
#
# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation,
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator=rf,
#                                param_distributions= random_grid,
#                                n_iter=100, cv=10,
#                                verbose=3, random_state=42,
#                                n_jobs=-1)  # Fit the random search model

# predictor matrix
X = np.asarray(train_data)[:, 0:42]
# response vector
y = np.asarray(train_data)[:, 42]

# model = rf_random.fit(X, y)
# rf_random.best_params_
# best_random = rf_random.best_estimator_
#
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [25, 30, 35, 40],
#     'max_features': [2, 4, 6, 8, 10],
#     'min_samples_leaf': [2, 4, 6, 8],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [130, 140, 150]
# }
#
# # Create a based model
# rf = RandomForestClassifier()  # Instantiate the grid search model
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
#                            cv=10, n_jobs=-1, verbose=3)
#
# # Fit the grid search to the data
# grid_search.fit(X, y)
# grid_search.best_params_

# specify model hyper-parameters
clf = RandomForestClassifier(
    max_depth=30,
    max_features='sqrt',
    min_samples_leaf=8,
    min_samples_split=10,
    n_estimators=140,
    bootstrap=False,
    random_state=1,
    n_jobs=-1
)

# fit random forest
model = clf.fit(X, y)
score = model.score(X, y)

# save model
path = f'{workdir}/'
if not Path(path).exists():
    Path(path).mkdir(parents=True, exist_ok=True)
pkl_filename = f'{path}/rf_blue.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# ======================================================================================================================
# (4) predict images using the trained rf
# ======================================================================================================================

# list files to process
# dir = "Z:/Public/Jonas/004_Divers/Test2/Test_images/Orig"  # predict training images
dir = "Z:/Public/Jonas/004_Divers/Test2/Evaluation/orig"  # predict all images
files_orig = glob.glob(f'{dir}/*.jpg')

# dir = "Z:/Public/Jonas/004_Divers/Test2/Test_images/BGR"  # predict training images
dir = "Z:/Public/Jonas/004_Divers/Test2/Evaluation/pp"  # predict all images
files_pp = glob.glob(f'{dir}/*.jpg')

workdir = "Z:/Public/Jonas/004_Divers/Test2"
colors = ["green"]
for color in colors:

    print(color)
    path = f'{workdir}/{color}/rf_{color}.pkl'
    with open(path, 'rb') as model:
        model = pickle.load(model)

    for file in files_orig:

        # get original image
        print(file)
        bname = os.path.splitext(os.path.basename(file))[0]
        img = imageio.imread(file)
        img = img[:, :, :3]
        img = np.ascontiguousarray(img, dtype=np.uint8)

        color_spaces_orig, descriptors_orig, descriptor_names_orig = SegmentationFunctions.get_color_spaces(img)
        descriptor_names_orig = [sub + "_orig" for sub in descriptor_names_orig]

        # get pre-processed image
        # file_pp = file.replace("Orig", "BGR")  # < ===== !!!!!!!!!!!!!!!!!!!!!!!!!!!
        dir_pp = "Z:/Public/Jonas/004_Divers/Test2/Evaluation/pp"
        filename_pp = os.path.basename(file).replace(".jpg", "")
        try:
            file_path_pp = glob.glob(f'{dir_pp}/{filename_pp}*.jpg')[0]
        except IndexError:
            continue

        img_pp = imageio.imread(file_path_pp)

        color_spaces_pp, descriptors_pp, descriptor_names_pp = SegmentationFunctions.get_color_spaces(img_pp)
        descriptor_names_pp = [sub + "_pp" for sub in descriptor_names_pp]

        descriptors = np.concatenate((descriptors_orig, descriptors_pp), axis=2)
        descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

        a_segmented_flatten = model.predict(descriptors_flatten)

        a_segmented = a_segmented_flatten.reshape((descriptors.shape[0], descriptors.shape[1]))

        a_segmented = np.where(a_segmented == 'pos', 255, 0)
        a_segmented = a_segmented.astype("uint8")

        # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs[0].imshow(img)
        # axs[0].set_title('original patch')
        # axs[1].imshow(img_pp)
        # axs[1].set_title('filtered patch')
        # axs[2].imshow(a_segmented)
        # axs[2].set_title('segmentation')
        # plt.show(block=True)

        imageio.imwrite(f'{workdir}/Evaluation/predicted/{color}/{bname}.tiff', a_segmented)


# ======================================================================================================================
# POST-PROCESS MASKS
# extract leaf
# ======================================================================================================================

from skimage import morphology
import math
import copy
from skimage.filters.rank import entropy as Entropy


# get image names
workdir = "P:/Public/Jonas/004_Divers/Test2/Evaluation"
image_paths = glob.glob(f'{workdir}/orig/*.jpg')
image_names = []
for p in image_paths:
    image_name = os.path.splitext(os.path.basename(p))[0]
    image_names.append(image_name)

for image_name in image_names:

    print(image_name)

    # load image and all masks
    image = imageio.imread(f'{workdir}/orig/{image_name}.jpg')

    # convert pixels to gray scale
    graypatch = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # get entropy as texture measure
    graypatch_sm = cv2.medianBlur(graypatch, 31)
    img_ent = Entropy(graypatch_sm, morphology.disk(10))

    # threshold on texture
    mask = np.where(img_ent > 1.9, 1, 0)

    # clean mask
    # add boundary (x)
    min_x_1 = np.min(np.where(mask[:, 0] != 0))
    max_x_1 = np.max(np.where(mask[:, 0] != 0))
    min_x_2 = np.min(np.where(mask[:, -1] != 0))
    max_x_2 = np.max(np.where(mask[:, -1] != 0))
    mask[min_x_1:max_x_1:, 0] = 1
    mask[min_x_2:max_x_2:, -1] = 1

    # add boundary (y)
    try:
        min_x_1 = np.min(np.where(mask[0, :] != 0))
        max_x_1 = np.max(np.where(mask[0, :] != 0))
        mask[0, min_x_1:max_x_1] = 1
    except ValueError:
        mask = mask
    try:
        min_x_2 = np.min(np.where(mask[-1, :] != 0))
        max_x_2 = np.max(np.where(mask[-1, :] != 0))
        mask[-1, min_x_2:max_x_2] = 1
    except ValueError:
        mask = mask

    # fill holes
    out2 = ndi.binary_fill_holes(mask).astype("uint8")*255

    kernel = np.ones((17, 17), np.uint8)
    out2 = morphology.erosion(out2, kernel)
    # plt.imshow(out2)

    # detect leaf/leaves (remove background)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([45, 70, 70], dtype=np.uint8)
    mask_leaf = cv2.inRange(image, lower_black, upper_black)  # could also use threshold
    mask_leaf = cv2.medianBlur(mask_leaf, 15)
    mask_leaf = utils.filter_objects_size(mask_leaf, size_th=7500, dir="smaller")
    mask_leaf_inv = 255-mask_leaf
    mask_leaf_inv = utils.filter_objects_size(mask_leaf_inv, size_th=400, dir="smaller")
    mask_leaf = 255 - mask_leaf_inv

    mask_leaf_inv = 255-mask_leaf

    final_mask = np.bitwise_and(out2, mask_leaf_inv)

    # save leaf mask
    imageio.imwrite(f'{workdir}/leaf_mask/{image_name}.tiff', out2)

    # check image
    check_img = copy.copy(image)

    _, cnt, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in cnt:
        cv2.drawContours(check_img, [c], 0, (255, 0, 0), 2)

    # save overlay
    imageio.imwrite(f'{workdir}/leaf_mask/overlay/{image_name}.tiff', check_img)

# ======================================================================================================================
# POST-PROCESS MASKS
# filter masks
# ======================================================================================================================

for image_name in image_names:

    print(image_name)

    # get pycnidia masks
    mask_blue = imageio.imread(f'{workdir}/predicted/blue/{image_name}.tiff')
    mask_green = imageio.imread(f'{workdir}/predicted/green/{image_name}.tiff')
    mask_red = imageio.imread(f'{workdir}/predicted/red/{image_name}.tiff')
    masks = [mask_red, mask_green, mask_blue]

    # get leaf mask and overlay
    leaf_mask = imageio.imread(f'{workdir}/leaf_mask/{image_name}.tiff')
    leaf_overlay = imageio.imread(f'{workdir}/leaf_mask/overlay/{image_name}.tiff')

    # get image
    image = imageio.imread(f'{workdir}/orig/{image_name}.jpg')

    # ==================================================================================================================

    # post-process masks and create overlay
    combined_mask = np.zeros(image.shape[:2], np.uint8)
    i = 0
    for m in masks:
        # crop leaves
        cropped = m * leaf_mask

        # filter objects
        # filter by size
        mask_blur = cv2.medianBlur(cropped, 9)  # de-noise

        # color-dependent size threshold
        if i == 0:
            mask_filt = utils.filter_objects_size(mask_blur, size_th=19, dir="smaller")
        elif i == 1:
            mask_filt = utils.filter_objects_size(mask_blur, size_th=13, dir="smaller")
        elif i == 2:
            mask_filt = utils.filter_objects_size(mask_blur, size_th=5, dir="smaller")

        mask_filt = utils.filter_objects_size(mask_filt, size_th=1200, dir="greater")

        # find object contours
        _, contours, _ = cv2.findContours(mask_filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # check if contour is of circular shape
        # only for red
        if i == 0:
            contours_circles = []
            for c in contours:
                perimeter = cv2.arcLength(c, True)
                area = cv2.contourArea(c)
                if perimeter == 0:
                    break
                circularity = 4*math.pi*(area/(perimeter*perimeter))
                if 0.6 < circularity:
                    contours_circles.append(c)
            contours = contours_circles

        # # create overlay
        # for contour in contours:
        #     cv2.drawContours(leaf_overlay, [contour], 0, (255, 255, 255), 2)

        if i == 0:
            color = (0, 255, 0)
        elif i == 1:
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)
        for contour in contours:
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.drawMarker(leaf_overlay, (cx, cy), color, markerType=cv2.MARKER_CROSS,
                           markerSize=6, thickness=2, line_type=cv2.LINE_AA)

        # create overlay
        for contour in contours:
            cv2.drawContours(combined_mask, [contour], 0, i+1, -1)

        i += 1

    # save resulting image
    imageio.imwrite(f'{workdir}/processed/combined_mask/{image_name}.tiff', combined_mask)
    imageio.imwrite(f'{workdir}/processed/overlay/{image_name}.tiff', leaf_overlay)

#
# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# # Show RGB and segmentation mask
# axs[0].imshow(image)
# axs[0].set_title('original patch')
# axs[1].imshow(out4)
# axs[1].set_title('original patch')
# plt.show(block=True)
#
# fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
# # Show RGB and segmentation mask
# axs[0, 0].imshow(cropped[0])
# axs[0, 0].set_title('red')
# axs[1, 0].imshow(cropped[1])
# axs[1, 0].set_title('green')
# axs[0, 1].imshow(cropped[2])
# axs[0, 1].set_title('blue')
# axs[1, 1].imshow(test_img)
# axs[1, 1].set_title('original')
# plt.show(block=True)


# OLD: LEAF EXTRACTION

    # # detect leaf/leaves (remove background)
    # lower_black = np.array([0, 0, 0], dtype=np.uint8)
    # upper_black = np.array([40, 75, 75], dtype=np.uint8)
    # mask_leaf = cv2.inRange(image, lower_black, upper_black)  # could also use threshold
    # # mask_leaf = cv2.medianBlur(mask_leaf, 35)
    #
    # # invert to get the leaf
    # mask_leaf = cv2.bitwise_not(mask_leaf)
    # mask_leaf = cv2.medianBlur(mask_leaf, 35)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    # mask_leaf = morphology.erosion(mask_leaf, kernel)
    #
    # # filter by size
    # out = utils.filter_objects_size(mask_leaf, size_th=100000, dir="smaller")
    #
    # # fill holes and erode to remove border regions
    # out2 = ndi.binary_fill_holes(out).astype("uint8")*255
    #
    # # invert to get background
    # out_neg = cv2.bitwise_not(out2)
    # out3 = utils.filter_objects_size(out_neg, size_th=10000, dir="smaller")
    #
    # out_pos = cv2.bitwise_not(out3)
    # final = cv2.medianBlur(out_pos, 35)
    # # erode to get rid of leaf edges
    # final = utils.filter_objects_size(final, size_th=100000, dir="smaller")
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    # final = morphology.erosion(final, kernel)
    # _, cnt, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # test_img = copy.copy(image)
    # for c in cnt:
    #     cv2.drawContours(test_img, [c], 0, (255, 0, 0), 2)
    #
    # # save leaf mask
    # imageio.imwrite(f'{workdir}/Processed/Leaf_masks/{image_name}.tiff', final)


# ======================================================================================================================
# Identify segments
# ======================================================================================================================

from skimage.segmentation import watershed
from skimage import morphology

# get image names
workdir = "P:/Public/Jonas/004_Divers/PycnidiaPattern_all"
image_paths = glob.glob(f'{workdir}/*.tif')
image_names = []
for p in image_paths:
    image_name = os.path.splitext(os.path.basename(p))[0]
    image_names.append(image_name)

indices = [i for i, s in enumerate(image_names) if 'MIX_P1_L4_im5' in s]

logfile = pd.read_csv(f'{workdir}/processed/overlay2/logfile.csv')

files_analyze = []
for index, row in logfile.iterrows():
    if row["action"] == "None":
        base_name = os.path.basename(row["path"])
        file_name = base_name.replace(".tiff", "")
        files_analyze.append(file_name)
    else:
        continue

iterator = 0
for image_name in files_analyze:

    iterator += 1
    print(f'Processing {iterator}/{len(files_analyze)}')
    check = imageio.imread(f'{workdir}/processed/combined_mask/{image_name}.tiff')
    leaf_mask = imageio.imread(f'{workdir}/leaf_mask/{image_name}.tiff')
    leaf_mask = np.where(leaf_mask != 0, 1, leaf_mask)
    orig_image = imageio.imread(f'{workdir}/JPG_Files/{image_name}.jpg')
    raw_image = imageio.imread(f'{workdir}/{image_name}.tif')

    # plt.imshow(check)
    # plt.imshow(leaf_mask)
    # plt.imshow(orig_image)

    # distance
    mask_bin = np.where(check != 0, 1, check)
    mask_bin_inv = np.where(mask_bin == 0, 1, 0)
    distance = ndi.distance_transform_edt(mask_bin_inv)
    check = np.where(distance > 100, 4, check)

    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # # Show RGB and segmentation mask
    # axs[0].imshow(orig_image)
    # axs[0].set_title('original patch')
    # axs[1].imshow(distance)
    # axs[1].set_title('original patch')
    # plt.show(block=True)

    # binarize the post-processed mask
    mask = np.where(check != 0, 255, check)
    # component labelling
    n_comps, output, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # invert the mask
    mask_inv = np.bitwise_not(mask)
    # calculate the euclidian distance transform
    distance = ndi.distance_transform_edt(mask_inv)

    labels = watershed(distance, markers=check, watershed_line=True, compactness=0)
    # erosion to thicken watershed boundaries
    kernel = np.ones((2, 2), np.uint8)
    labels_erode = morphology.erosion(labels, kernel)
    labels_out = leaf_mask * labels_erode

    # plt.imshow(labels_out)

    area_leaf = len(np.where(leaf_mask != 0)[0])

    # count clusters (not including the "empty" ones)
    # clusters = np.where(labels_out == 4, 0, labels_out)   # remove empty
    clusters = labels_out
    cl_bin = np.where(clusters != 0, 255, 0).astype("uint8")  # make binary
    n_comps, output, _, _ = cv2.connectedComponentsWithStats(cl_bin, connectivity=8)  # label objects
    n_segments = n_comps - 1  # remove background
    n_segments_pyc = n_segments

    id = []
    number = []
    sizes = []
    for i in range(1, n_comps):
        m = np.where(output == i, 1, 0).astype("uint8")
        m_p = (np.bitwise_and(m, mask_bin))
        n_comp, out, stats, _ = cv2.connectedComponentsWithStats(m_p, connectivity=8)  # label objects
        n_comp = n_comp - 1
        # check if there are pycnidia in the segment
        if n_comp == 0:
            n_segments_pyc = n_segments_pyc - 1  # remove segment from total count
        try:
            type_id = np.unique(np.where(output == i, check, 0))[1]
        except IndexError:
            type_id = np.unique(np.where(output == i, check, 0))[0]
        id.append(type_id)
        number.append(n_comp)
        _, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)  # label objects
        sizes.append(stats[1:, -1][0])
    df = pd.DataFrame({'color_id': id, 'n_pyc': number, 'size_seg': sizes, 'leaf_area': area_leaf,
                       'n_segments': n_segments, 'n_segments_pyc': n_segments_pyc})

    df.to_csv(f'{workdir}/Output_segments/{image_name}.csv', index=False)

    # save resulting image
    output_mask_all = output*255.0 / output.max().astype("uint8")
    imageio.imwrite(f'{workdir}/Output_segments/Segments/all/{image_name}.png', output_mask_all)

    output_mask_pyc = labels_out*255.0 / labels_out.max().astype("uint8")
    imageio.imwrite(f'{workdir}/Output_segments/Segments/pyc/{image_name}.png', output_mask_pyc)

    maskout = np.where(labels_out == 0)
    orig_image[maskout] = (0, 0, 0)
    imageio.imwrite(f'{workdir}/Output_segments/Overlay/{image_name}.tiff', orig_image)





fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# Show RGB and segmentation mask
axs[0].imshow(orig_image)
axs[0].set_title('original patch')
axs[1].imshow(check)
axs[1].set_title('original patch')
plt.show(block=True)



image = imageio.imread(f'{workdir}/Test_images/{image_name}.jpg')
sd = cv2.meanStdDev(image, mask=leaf_mask)

R, G, B = cv2.split(image)

# img = cv2.merge([R, G, B])
# plt.imshow(img)

# subtract mean
R = R.astype(float)-171
G = G.astype(float)-112
B = B.astype(float)-104

R = R/56
G = G/40
B = B/39

R = (R - np.min(R))/np.ptp(R)*255
G = (G - np.min(G))/np.ptp(G)*255
B = (B - np.min(B))/np.ptp(B)*255

img_rr = cv2.merge([R.astype("uint8"), G.astype("uint8"), B.astype("uint8")])

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# Show RGB and segmentation mask
axs[0].imshow(image)
axs[0].set_title('original patch')
axs[1].imshow(img_rr)
axs[1].set_title('original patch')
plt.show(block=True)
