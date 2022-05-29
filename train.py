import torch.utils.data as data
from tqdm import tqdm
from dataset import RoadExtractionDataset
from extractor import Extractor

# # DLinkNet
# from network import DLinkNet34
# from loss import DiceBCELoss
#
# root_dir = '../datasets/deepglobe/train_crops/'
# weight_dir = '../weights/essentialfeatures/dlinknet/'
#
# dataset = RoadExtractionDataset(root_dir)
# dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True)
#
# model = Extractor(DLinkNet34, DiceBCELoss)
#
# total_epoch = 300
# train_epoch_best_loss = 100.
# no_optim = 0
# for epoch in range(0, total_epoch + 1):
#     dataloader_iter = iter(dataloader)
#     train_epoch_loss = 0
#     for image, mask, _ in tqdm(dataloader_iter):
#         train_loss = model.optimize()
#         train_epoch_loss += train_loss
#     train_epoch_loss /= len(dataloader)
#     print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.old_lr)
#     if train_epoch_loss >= train_epoch_best_loss:
#         no_optim += 1
#     else:
#         no_optim = 0
#         model.save(weight_dir + str(epoch) + '.th')
#         model.save(weight_dir + 'best.th')
#         train_epoch_best_loss = train_epoch_loss
#     if no_optim > 6:
#         print('early stop at %d epoch' % epoch)
#         break
#     if no_optim > 3:
#         if model.old_lr < 5e-7:
#             break
#         model.load(weight_dir + 'best.th')
#         model.update_lr()

# # DLinkNet + HT
# from network import HTDLinkNet34
# from loss import DiceBCELoss
#
# root_dir = '../datasets/deepglobe/train_crops/'
# weight_dir = '../weights/essentialfeatures/htdlinknet/'
#
# dataset = RoadExtractionDataset(root_dir)
# dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True)
#
# model = Extractor(HTDLinkNet34, DiceBCELoss)
#
# total_epoch = 300
# train_epoch_best_loss = 100.
# no_optim = 0
# for epoch in range(0, total_epoch + 1):
#     dataloader_iter = iter(dataloader)
#     train_epoch_loss = 0
#     for image, mask, _ in tqdm(dataloader_iter):
#         train_loss = model.optimize()
#         train_epoch_loss += train_loss
#     train_epoch_loss /= len(dataloader)
#     print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.old_lr)
#     if train_epoch_loss >= train_epoch_best_loss:
#         no_optim += 1
#     else:
#         no_optim = 0
#         model.save(weight_dir + str(epoch) + '.th')
#         model.save(weight_dir + 'best.th')
#         train_epoch_best_loss = train_epoch_loss
#     if no_optim > 6:
#         print('early stop at %d epoch' % epoch)
#         break
#     if no_optim > 3:
#         if model.old_lr < 5e-7:
#             break
#         model.load(weight_dir + 'best.th')
#         model.update_lr()

# # DLinkNet + NR
# from network import MDLinkNet34
# from loss import SmoothDiceBCELoss
#
# root_dir = '../datasets/deepglobe/train_crops/'
# weight_dir = '../weights/essentialfeatures/mdlinknet/'
#
# dataset = RoadExtractionDataset(root_dir)
# dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True)
#
# model = Extractor(MDLinkNet34, SmoothDiceBCELoss)
#
# total_epoch = 300
# train_epoch_best_loss = 100.
# no_optim = 0
# for epoch in range(0, total_epoch + 1):
#     dataloader_iter = iter(dataloader)
#     train_epoch_loss = 0
#     for image, mask, nr_map in tqdm(dataloader_iter):
#         model.set_input(image, mask, nr_map)
#         train_loss = model.optimize()
#         train_epoch_loss += train_loss
#     train_epoch_loss /= len(dataloader)
#     print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.old_lr)
#     if train_epoch_loss >= train_epoch_best_loss:
#         no_optim += 1
#     else:
#         no_optim = 0
#         model.save(weight_dir + str(epoch) + '.th')
#         model.save(weight_dir + 'best.th')
#         train_epoch_best_loss = train_epoch_loss
#     if no_optim > 6:
#         print('early stop at %d epoch' % epoch)
#         break
#     if no_optim > 3:
#         if model.old_lr < 5e-7:
#             break
#         model.load(weight_dir + 'best.th')
#         model.update_lr()

# DLinkNet + HT + NR
from network import MHTDLinkNet34
from loss import SmoothDiceBCELoss

root_dir = '../datasets/deepglobe/train_crops/'
weight_dir = '../weights/essentialfeatures/mhtdlinknet/'

dataset = RoadExtractionDataset(root_dir)
dataloader = data.DataLoader(dataset, batch_size=12, shuffle=True)

model = Extractor(MHTDLinkNet34, SmoothDiceBCELoss)
model.load(weight_dir + 'best.th')

total_epoch = 300
train_epoch_best_loss = 100.
no_optim = 0
for epoch in range(59, total_epoch + 1):
    dataloader_iter = iter(dataloader)
    train_epoch_loss = 0
    for image, mask, nr_map in tqdm(dataloader_iter):
        model.set_input(image, mask, nr_map)
        train_loss = model.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(dataloader)
    print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.old_lr)
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        model.save(weight_dir + str(epoch) + '.th')
        model.save(weight_dir + 'best.th')
        train_epoch_best_loss = train_epoch_loss
    if no_optim > 6:
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if model.old_lr < 5e-7:
            break
        model.load(weight_dir + 'best.th')
        model.update_lr()

# # ResUNet
# from network import ResUNet
# from loss import DiceBCELoss
#
# root_dir = '../datasets/deepglobe/train_crops/'
# weight_dir = '../weights/essentialfeatures/resunet/'
#
# dataset = RoadExtractionDataset(root_dir, nr_head=False)
# dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True)
#
# model = Extractor(ResUNet, DiceBCELoss)
#
# total_epoch = 300
# train_epoch_best_loss = 100.
# no_optim = 0
# for epoch in range(1, total_epoch + 1):
#     dataloader_iter = iter(dataloader)
#     train_epoch_loss = 0
#     for image, mask in tqdm(dataloader_iter):
#         model.set_input(image, mask)
#         train_loss = model.optimize()
#         train_epoch_loss += train_loss
#     train_epoch_loss /= len(dataloader)
#     print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.old_lr)
#     if train_epoch_loss >= train_epoch_best_loss:
#         no_optim += 1
#     else:
#         no_optim = 0
#         model.save(weight_dir + str(epoch) + '.th')
#         model.save(weight_dir + 'best.th')
#         train_epoch_best_loss = train_epoch_loss
#     if no_optim > 6:
#         print('early stop at %d epoch' % epoch)
#         break
#     if no_optim > 3:
#         if model.old_lr < 5e-7:
#             break
#         model.load(weight_dir + 'best.th')
#         model.update_lr()

# # UNet
# from network import UNet
# from loss import DiceBCELoss
#
# root_dir = '../datasets/deepglobe/train_crops/'
# weight_dir = '../weights/essentialfeatures/unet/'
#
# dataset = RoadExtractionDataset(root_dir, nr_head=False)
# dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True)
#
# model = Extractor(UNet, DiceBCELoss)
#
# total_epoch = 300
# train_epoch_best_loss = 100.
# no_optim = 0
# for epoch in range(1, total_epoch + 1):
#     dataloader_iter = iter(dataloader)
#     train_epoch_loss = 0
#     for image, mask, _ in tqdm(dataloader_iter):
#         train_loss = model.optimize()
#         train_epoch_loss += train_loss
#     train_epoch_loss /= len(dataloader)
#     print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.old_lr)
#     if train_epoch_loss >= train_epoch_best_loss:
#         no_optim += 1
#     else:
#         no_optim = 0
#         model.save(weight_dir + str(epoch) + '.th')
#         model.save(weight_dir + 'best.th')
#         train_epoch_best_loss = train_epoch_loss
#     if no_optim > 6:
#         print('early stop at %d epoch' % epoch)
#         break
#     if no_optim > 3:
#         if model.old_lr < 5e-7:
#             break
#         model.load(weight_dir + 'best.th')
#         model.update_lr()
