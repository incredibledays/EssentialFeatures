import torch.utils.data as data
import cv2
import numpy as np
import tqdm
from dataset import RoadExtractionDataset
from extractor import Extractor


# # DLinkNet
# from network import DLinkNet34
#
# root_dir = '../datasets/deepglobe/val_crops/'
# weight_dir = '../weights/essentialfeatures/dlinknet/best.th'
# result_dir = '../results/essentialfeatures/dlinknet/'
#
# dataset = RoadExtractionDataset(root_dir, is_train=False, nr_head=False)
# dataloader = data.DataLoader(dataset)
#
# model = Extractor(DLinkNet34, eval_mode=True)
# model.load(weight_dir)
#
# dataloader_iter = iter(dataloader)
# for image, name in tqdm.tqdm(dataloader_iter):
#     model.set_input(image)
#     pred = model.predict()
#     pred[pred > 0.5] = 255
#     pred[pred <= 0.5] = 0
#     cv2.imwrite(result_dir + name[0] + '.png', pred.astype(np.uint8))

# # HTDLinkNet
# from network import HTDLinkNet34
#
# root_dir = '../datasets/deepglobe/val_crops/'
# weight_dir = '../weights/essentialfeatures/htdlinknet/best.th'
# result_dir = '../results/essentialfeatures/htdlinknet/'
#
# dataset = RoadExtractionDataset(root_dir, is_train=False, nr_head=False)
# dataloader = data.DataLoader(dataset)
#
# model = Extractor(HTDLinkNet34, eval_mode=True)
# model.load(weight_dir)
#
# dataloader_iter = iter(dataloader)
# for image, name in tqdm.tqdm(dataloader_iter):
#     model.set_input(image)
#     pred = model.predict()
#     pred[pred > 0.5] = 255
#     pred[pred <= 0.5] = 0
#     cv2.imwrite(result_dir + name[0] + '.png', pred.astype(np.uint8))

# MHTDLinkNet
from network import MHTDLinkNet34

root_dir = '../datasets/deepglobe/val_crops/'
weight_dir = '../weights/essentialfeatures/mhtdlinknet/best.th'
result_dir = '../results/essentialfeatures/mhtdlinknet/'

dataset = RoadExtractionDataset(root_dir, is_train=False, nr_head=False)
dataloader = data.DataLoader(dataset)

model = Extractor(MHTDLinkNet34, eval_mode=True)
model.load(weight_dir)

dataloader_iter = iter(dataloader)
for image, name in tqdm.tqdm(dataloader_iter):
    model.set_input(image)
    pred = model.predict(True)
    pred[pred > 0.5] = 255
    pred[pred <= 0.5] = 0
    cv2.imwrite(result_dir + name[0] + '.png', pred.astype(np.uint8))
