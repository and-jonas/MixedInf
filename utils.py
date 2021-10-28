import imageio
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from skimage import morphology
import math
import copy
from scipy.spatial.distance import cdist
import utils
import os
from importlib import reload
reload(utils)
from matplotlib import pyplot as plt
import scipy.interpolate as si


def keep_largest_object(mask):
    """
    Filter objects in a binary mask to keep only the largest object
    :param mask: A binary mask to filter
    :return: A binary mask containing only the largest object of the original mask
    """
    _, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ctr = np.around(centroids)[1:].astype("int")
    sizes = list(stats[:, 4][1:])
    index = sizes.index(np.max(sizes))
    cleaned = np.uint8(np.where(output == index + 1, 1, 0))

    return cleaned


def filter_objects_size(mask, size_th, dir):
    """
    Filter objects in a binary mask by size
    :param mask: A binary mask to filter
    :param size_th: The size threshold used to filter (objects GREATER than the threshold will be kept)
    :return: A binary mask containing only objects greater than the specified threshold
    """
    _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    if dir == "greater":
        idx = (np.where(sizes > size_th)[0] + 1).tolist()
    if dir == "smaller":
        idx = (np.where(sizes < size_th)[0] + 1).tolist()
    out = np.in1d(output, idx).reshape(output.shape)
    cleaned = np.where(out, 0, mask)

    return cleaned


def filter_by_distance(mask, min_distance=15):
    """
    Filter objects in a binary mask by centroid position
    :param mask: A binary mask to filter
    :return: Coordinates of filtered centroids
    """
    n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if n_comps == 1:
        ctrs_ok = centroids[1:]
    else:
        centroids = centroids[1:]
        dist = cdist(centroids, centroids)
        idx = np.where((dist > 0) & (dist < min_distance))[1].tolist()

        ctrs = centroids[idx]

        ctrs_ok = np.delete(centroids, idx, axis=0)

        data = np.split(ctrs, list(range(2, 4, 2)))

        for sample in data:
            ctr_corr = np.mean(sample, axis=0)
            ctrs_ok = np.append(ctrs_ok, [ctr_corr], axis=0)

    return ctrs_ok


def preprocess_image(image):

    sd = cv2.meanStdDev(image)

    R, G, B = cv2.split(image)

    # subtract mean
    R = R.astype(float) - sd[0][0]
    G = G.astype(float) - sd[0][1]
    B = B.astype(float) - sd[0][2]

    R = R / sd[1][0]
    G = G / sd[1][1]
    B = B / sd[1][2]

    R = (R - np.min(R)) / np.ptp(R) * 255
    G = (G - np.min(G)) / np.ptp(G) * 255
    B = (B - np.min(B)) / np.ptp(B) * 255

    img_rr = cv2.merge([R.astype("uint8"), G.astype("uint8"), B.astype("uint8")])

    return img_rr


def normalize_image_illumination(image):

    hh, ww = image.shape[:2]
    print(hh, ww)
    maximum = max(hh, ww)

    # illumination normalize
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # separate channels
    y, cr, cb = cv2.split(ycrcb)

    # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
    # account for size of input vs 300
    sigma = int(5 * maximum / 300)
    gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)

    # subtract background from Y channel
    y = (y - gaussian + 100)

    # merge channels back
    ycrcb = cv2.merge([y, cr, cb])

    # convert to BGR
    output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return output

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # Show RGB and segmentation mask
    axs[0].imshow(image)
    axs[0].set_title('original patch')
    axs[1].imshow(gaussian)
    axs[1].set_title('original patch')
    plt.show(block=True)


# def post_process_mask(img, mask_red, mask_green, mask_blue):
#
#     # detect leaf/leaves (remove background)
#     lower_black = np.array([0, 0, 0], dtype=np.uint8)
#     upper_black = np.array([40, 40, 40], dtype=np.uint8)
#     mask_leaf = cv2.inRange(img, lower_black, upper_black)  # could also use threshold
#     mask_leaf = cv2.bitwise_not(mask_leaf)
#     out = utils.filter_objects_size(mask_leaf, size_th=10000, dir="smaller")
#
#     # fill holes and erode to remove border regions
#     out2 = ndi.binary_fill_holes(out).astype("uint8")
#     kernel = np.ones((25, 25), np.uint8)
#     out3 = morphology.dilation(out2, kernel)
#     kernel = np.ones((50, 50), np.uint8)
#     out4 = morphology.erosion(out3, kernel)
#     mask_leaf = mask * out4
#
#     # filter objects
#     # filter by size
#     mask_blur = cv2.medianBlur(mask_leaf, 5)  # de-noise
#     mask_filt = utils.filter_objects_size(mask_blur, size_th=15, dir="smaller")
#     mask_filt = utils.filter_objects_size(mask_filt, size_th=1000, dir="greater")
#
#     # check if contour is of circular shape
#     _, contours, _ = cv2.findContours(mask_filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours_circles = []
#     for c in contours:
#         perimeter = cv2.arcLength(c, True)
#         area = cv2.contourArea(c)
#         if perimeter == 0:
#             break
#         circularity = 4 * math.pi * (area / (perimeter * perimeter))
#         print(circularity)
#         if 0.7 < circularity:
#             contours_circles.append(c)
#
#     # create overlay
#     test_img = copy.copy(img)
#     for contour in contours_circles:
#         cv2.drawContours(test_img, [contour], 0, (0, 255, 0), 2)


def process_image(path_to_image):

    # get file name
    basename = os.path.basename(path_to_image)
    # load image
    image = imageio.imread(path_to_image)
    # blur image
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    # transform to HSV
    imhsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    # GREEN ============================================================================================================

    # minimum red amount, max red amount
    min_green = np.array([32, 70, 80])
    max_green = np.array([75, 255, 255])
    # layer
    mask = cv2.inRange(imhsv, min_green, max_green)

    # filter by size
    mask_filtered = filter_objects_size(mask, 8, "smaller")

    # filter by distance
    ctr_coordinates = filter_by_distance(mask_filtered, min_distance=15)

    # draw onto image
    coord = pd.DataFrame(ctr_coordinates)
    test_img = copy.copy(image)
    for index, row in coord.iterrows():
        coords = tuple([int(row[0]), int(row[1])])
        cv2.circle(test_img, coords, 5, (0, 0, 255), 2)

    # BLUE =============================================================================================================

    hsv_l = np.array([80, 100, 20])
    hsv_h = np.array([155, 255, 255])
    mask1 = cv2.inRange(imhsv, hsv_l, hsv_h)

    _, output, _, centroids_blue = cv2.connectedComponentsWithStats(mask1, connectivity=8)
    centroids_blue = centroids_blue[1:]

    # filter by distance
    ctr_coordinates = filter_by_distance(mask1, min_distance=15)

    # draw onto image
    coord = pd.DataFrame(ctr_coordinates)
    for index, row in coord.iterrows():
        coords = tuple([int(row[0]), int(row[1])])
        cv2.circle(test_img, coords, 5, (255, 0, 0), 2)

    # RED ==============================================================================================================

    # 1) DETECT CLEAR RED MAXIMA =======================================================================================

    lower_red = np.array([200, 0, 0])
    # lower_red = np.array([130,0 , 0])
    upper_red = np.array([255, 110, 110])

    mask = cv2.inRange(image, lower_red, upper_red)

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    # Show RGB and segmentation mask
    axs[0].imshow(image)
    axs[0].set_title('original patch')
    axs[1].imshow(imhsv)
    axs[1].set_title('original patch')
    axs[2].imshow(mask)
    axs[2].set_title('original patch')
    plt.show(block=True)

    # SIZE FILTER
    mask_filtered = filter_objects_size(mask, 500, "greater")
    mask_filtered = filter_objects_size(mask_filtered, 100, "smaller")

    _, contours, _ = cv2.findContours(mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_circles = []
    # check if contour is of circular shape
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        if perimeter == 0:
            break
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        if 0.4 < circularity:
            contours_circles.append(c)

    for contour in contours_circles:
        cv2.drawContours(test_img, [contour], 0, (0, 255, 0), 2)

    # 2) DETECT WEAKER RED MAXIMA ======================================================================================
    # 2A) ISOLATE PYCNIDIA AS DARK AREAS ===============================================================================

    dark = np.array([0, 0, 0])  # example value
    bright = np.array([90, 80, 80])  # example value
    mask_help = cv2.inRange(image, dark, bright)
    mask_help = utils.filter_objects_size(mask_help, 2000, "greater")  # removes large dark areas
    mask_help = utils.filter_objects_size(mask_help, 25, "smaller")  # removes noise
    # fill where possible
    mask_help = ndi.binary_fill_holes(mask_help).astype("uint8")
    kernel = np.ones((2, 2), np.uint8)
    mask_help = morphology.dilation(mask_help, kernel)
    _, contours, _ = cv2.findContours(mask_help, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # filter based on curl
    contours_curl = []
    for c in contours:
        hull = cv2.convexHull(c, returnPoints=True)
        hull = np.vstack(hull).squeeze()
        # max pairwise distance
        distances = cdist(hull, hull)
        maxdist = np.max(distances)
        area = cv2.contourArea(c)
        if area > 1500:
            continue
        perimeter = cv2.arcLength(c, True)
        try:
            fibre_length = perimeter - math.sqrt(perimeter * perimeter - 16 * area) / 4
        except ValueError:
            continue
        curl = maxdist / fibre_length
        if curl < 0.4:
            contours_curl.append(c)

    # filter based on circularity
    contours_circles = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        if area < 150:
            continue
        if perimeter == 0:
            break
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        if 0.6 < circularity:
            contours_circles.append(c)

    # create mask and sub-image
    image_glance = copy.copy(image)
    empty_mask = np.zeros(image_glance.shape).astype("uint8")
    for contour in contours_curl:
        hull = cv2.convexHull(contour, returnPoints=True)
        cv2.drawContours(image_glance, [contour], 0, (0, 255, 0), 2)
        cv2.drawContours(empty_mask, [hull], 0, (255, 255, 255), -1)
    for contour in contours_circles:
        hull = cv2.convexHull(contour, returnPoints=True)
        cv2.drawContours(image_glance, [contour], 0, (0, 0, 255), 1)
        cv2.drawContours(empty_mask, [hull], 0, (255, 255, 255), -1)

    # 2B) FILTER FOR RED PYCNIDIA ON THE SUB-IMAGE =====================================================================

    kernel = np.ones((3, 3, 3), np.uint8)
    final_adj_mask = morphology.dilation(empty_mask, kernel)

    image_sub = cv2.bitwise_and(image, final_adj_mask)

    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 80, 80])
    mask = cv2.inRange(image_sub, lower_red, upper_red)

    kernel = np.ones((3, 3), np.uint8)
    mask = morphology.erosion(mask, kernel)
    mask_final = utils.filter_objects_size(mask, 25, "smaller")

    _, contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_additional = []
    # check if contour is of circular shape
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            break
        contours_additional.append(c)

    for contour in contours_additional:
        cv2.drawContours(test_img, [contour], 0, (0, 255, 0), 2)

    return test_img
