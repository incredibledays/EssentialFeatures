import torch
import torch.utils.data as data
import os
import cv2
import numpy as np


def random_hue_saturation_value(image, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255),
                                val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def random_shift_scale_rotate(image, mask, nr_map=None, shift_limit=(-0.0, 0.0), scale_limit=(-0.0, 0.0), rotate_limit=(-0.0, 0.0),
                              aspect_limit=(-0.0, 0.0), border_mode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode,
                                    borderValue=(0, 0, 0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode,
                                   borderValue=(0, 0, 0,))
        if nr_map is not None:
            nr_map = cv2.warpPerspective(nr_map, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode,
                                         borderValue=(0, 0, 0,))
            return image, mask, nr_map
        else:
            return image, mask
    else:
        if nr_map is not None:
            return image, mask, nr_map
        else:
            return image, mask


def random_horizontal_flip(image, mask, nr_map=None, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        if nr_map is not None:
            nr_map = cv2.flip(nr_map, 1)
            return image, mask, nr_map
        else:
            return image, mask
    else:
        if nr_map is not None:
            return image, mask, nr_map
        else:
            return image, mask


def random_vertical_flip(image, mask, nr_map=None, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        if nr_map is not None:
            nr_map = cv2.flip(nr_map, 0)
            return image, mask, nr_map
        else:
            return image, mask
    else:
        if nr_map is not None:
            return image, mask, nr_map
        else:
            return image, mask


def random_rotate_90(image, mask, nr_map=None, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)
        if nr_map is not None:
            nr_map = np.rot90(nr_map)
            return image, mask, nr_map
        else:
            return image, mask
    else:
        if nr_map is not None:
            return image, mask, nr_map
        else:
            return image, mask


class RoadExtractionDataset(data.Dataset):

    def __init__(self, root_dir, is_train=True, nr_head=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.nr_head = nr_head
        self.sample_list = list(map(lambda x: x[:-4], os.listdir(root_dir + 'images/')))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        if self.is_train:
            if self.nr_head:
                image = cv2.imread(os.path.join(self.root_dir, 'images/{}.png').format(self.sample_list[item]))
                mask = cv2.imread(os.path.join(self.root_dir, 'gt/{}.png').format(self.sample_list[item]),
                                  cv2.IMREAD_GRAYSCALE)
                nr_map = cv2.imread(os.path.join(self.root_dir, 'nr/{}.png').format(self.sample_list[item]))

                image = random_hue_saturation_value(image, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5),
                                                    val_shift_limit=(-15, 15))
                image, mask, nr_map = random_shift_scale_rotate(image, mask, nr_map, shift_limit=(-0.1, 0.1),
                                                                scale_limit=(-0.1, 0.1), aspect_limit=(-0.1, 0.1),
                                                                rotate_limit=(-0, 0))
                image, mask, nr_map = random_horizontal_flip(image, mask, nr_map)
                image, mask, nr_map = random_vertical_flip(image, mask, nr_map)
                image, mask, nr_map = random_rotate_90(image, mask, nr_map)

                image = np.array(image, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
                image = torch.Tensor(image)
                mask = np.expand_dims(mask, axis=2)
                mask = np.array(mask, np.float32).transpose((2, 0, 1)) / 255.0
                mask[mask >= 0.5] = 1
                mask[mask <= 0.5] = 0
                mask = torch.Tensor(mask)
                nr_map = np.array(nr_map, np.float32).transpose((2, 0, 1)) / 255.0
                nr_map = torch.Tensor(nr_map)
                return image, mask, nr_map
            else:
                image = cv2.imread(os.path.join(self.root_dir, 'images/{}.png').format(self.sample_list[item]))
                mask = cv2.imread(os.path.join(self.root_dir, 'gt/{}.png').format(self.sample_list[item]),
                                  cv2.IMREAD_GRAYSCALE)

                image = random_hue_saturation_value(image, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5),
                                                    val_shift_limit=(-15, 15))
                image, mask = random_shift_scale_rotate(image, mask, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1),
                                                        aspect_limit=(-0.1, 0.1), rotate_limit=(-0, 0))
                image, mask = random_horizontal_flip(image, mask)
                image, mask = random_vertical_flip(image, mask)
                image, mask = random_rotate_90(image, mask)

                image = np.array(image, np.float32).transpose((
                    2, 0, 1)) / 255.0 * 3.2 - 1.6
                image = torch.Tensor(image)
                mask = np.expand_dims(mask, axis=2)
                mask = np.array(mask, np.float32).transpose((2, 0, 1)) / 255.0
                mask[mask >= 0.5] = 1
                mask[mask <= 0.5] = 0
                mask = torch.Tensor(mask)
                return image, mask
        else:
            image = cv2.imread(os.path.join(self.root_dir, 'images/{}.png').format(self.sample_list[item]))
            image = np.array(image, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
            image = torch.Tensor(image)
            return image, self.sample_list[item]
