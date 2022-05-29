import os
from skimage import morphology
import cv2
from scipy import ndimage
import math
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    gt_root = '../datasets/deepglobe/train_crops/gt/'
    nr_root = '../datasets/deepglobe/train_crops/nr/'
    gt_list = os.listdir(gt_root)

    for gt in tqdm(gt_list):
        mask = cv2.imread(gt_root + gt, 0) > 128
        skel = morphology.skeletonize(mask)
        dist = ndimage.distance_transform_edt(~skel)
        grad_x = cv2.Sobel(dist, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(dist, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8
        dist_norm = np.expand_dims(np.uint8(dist / (512 * math.sqrt(2)) * 255), axis=2)
        grad_x_norm = np.expand_dims(np.uint8(grad_x / magnitude * 255), axis=2)
        grad_y_norm = np.expand_dims(np.uint8(grad_y / magnitude * 255), axis=2)
        nr_map = np.concatenate((dist_norm, grad_x_norm, grad_y_norm), axis=2)
        cv2.imwrite(nr_root + gt, nr_map)
