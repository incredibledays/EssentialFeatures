import os
import cv2
import numpy as np
import tqdm
from skimage import morphology

# result_path = '../results/essentialfeatures/resunet/'
# gt_path = '../datasets/deepglobe/val_crops/gt/'
#
# intersection = 0
# union = 0
# precision = 0
# recall = 0
# results = os.listdir(result_path)
# for result in tqdm.tqdm(results):
#     result_mask = cv2.imread(result_path + result, 0)
#     gt_mask = cv2.imread(gt_path + result, 0)
#     intersection += np.sum(np.array((result_mask == 255) & (gt_mask == 255)).astype(int))
#     union += np.sum(np.array((result_mask == 255) | (gt_mask == 255)).astype(int))
#     precision += np.sum(np.array(result_mask == 255).astype(int))
#     recall += np.sum(np.array(gt_mask == 255).astype(int))
#
# print('IOU: ', intersection / union)
# print('Precision: ', intersection / precision)
# print('Recall: ', intersection / recall)
# print('F1: ',
#       (2 * (intersection / precision) * (intersection / recall) / (intersection / precision + intersection / recall)))
#
# intersection = 0
# union = 0
# for result in tqdm.tqdm(results):
#     result_mask = cv2.imread(result_path + result, 0) > 128
#     gt_mask = cv2.imread(gt_path + result, 0) > 128
#     result_mask = morphology.skeletonize(result_mask)
#     gt_mask = morphology.skeletonize(gt_mask)
#     result_mask = morphology.dilation(result_mask, morphology.disk(4))
#     gt_mask = morphology.dilation(gt_mask, morphology.disk(4))
#     intersection += np.sum(np.array((result_mask == 1) & (gt_mask == 1)).astype(int))
#     union += np.sum(np.array((result_mask == 1) | (gt_mask == 1)).astype(int))
#
# print('IOU: ', intersection / union)


def get_relaxed_precision(a, b, buffer):
    tp = 0
    indices = np.where(a == 1)
    for ind in range(len(indices[0])):
        tp += (np.sum(
            b[indices[0][ind]-buffer: indices[0][ind]+buffer+1,
              indices[1][ind]-buffer: indices[1][ind]+buffer+1]) > 0).astype(np.int)
    return tp


pred_path = '../results/essentialfeatures/resunet/'
gt_path = '../datasets/deepglobe/val_crops/gt/'
files = os.listdir(pred_path)
rprecision_tp, rrecall_tp, pred_positive, gt_positive = 0, 0, 0, 0
for file in tqdm.tqdm(files):
    pred = cv2.imread(pred_path + file, 0) > 128
    gt = cv2.imread(gt_path + file, 0) > 128
    pred_sk = morphology.skeletonize(pred)
    gt_sk = morphology.skeletonize(gt)
    rprecision_tp += get_relaxed_precision(pred_sk, gt_sk, 4)
    rrecall_tp += get_relaxed_precision(gt_sk, pred_sk, 4)
    pred_positive += len(np.where(pred_sk == 1)[0])
    gt_positive += len(np.where(gt_sk == 1)[0])

precision = rprecision_tp/(gt_positive + 1e-12)
recall = rrecall_tp/(gt_positive + 1e-12)
f1measure = 2*precision*recall/(precision + recall + 1e-12)
iou = precision*recall/(precision+recall-(precision*recall) + 1e-12)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1: ', f1measure)
print('IOU: ', iou)
