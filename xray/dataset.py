import logging
import os
import re
import shelve

from PIL import Image

import xray.utils

import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision


class XRayDataset:
    def __init__(self, mode: str = 'train', data_dir: str = '../data/chest_xray/', split=0.8):
        self.mode = mode

        self.mode_dir = os.path.join(data_dir, self.mode if mode != 'eval' else 'train')

        self.available_files = [
            f.split('.')[0] for f in os.listdir(self.mode_dir) if f.endswith('dicom')
        ]

        if self.mode == 'train':
            self.available_files = self.available_files[:int(split * len(self.available_files))]
        elif self.mode == 'eval':
            self.available_files = self.available_files[int((1 -split) * len(self.available_files)):]

        self.logger = logging.getLogger(__name__)
        if self.mode != 'test':
            self.data_desc = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((400,400)),
            torchvision.transforms.ToTensor(),
            ZeroToOneTransform()
        ])


    def __getitem__(self, item, max_bboxes: int = 16):
        image = pydicom.read_file(os.path.join(self.mode_dir, self.available_files[item] + '.dicom'))
        if self.mode == 'test':
            return self.transform(image.pixel_array.copy().astype('float32'))
        file_description = self.data_desc.loc[self.data_desc.image_id == self.available_files[item]]

        # TODO: Do IoU for the bboxes and make some mean for shared boxes > than lets say 0.4
        class_names, bboxes, labels = [], [], []
        for _, row in file_description.iterrows():
            if not np.isnan(row.x_max):

                bboxes.append((row.x_min, row.y_min, row.x_max, row.y_max))
                labels.append(row.class_id + 1)
                class_names.append(row.class_name)
        bboxes, labels = tuple(map(torch.Tensor, [bboxes, labels]))

        assert len(bboxes) == len(labels) == len(class_names)

        bboxes_resized = torch.Tensor(
            list(map(lambda x: resize_bbox(x, image.pixel_array.shape, (400,400)), bboxes))
        )
        bboxes_resized, labels = xray.utils.filter_radiologist_findings(bboxes_resized, labels)

        target = {
            'boxes': bboxes_resized,
            'labels': labels.long(),
            'file_name': self.available_files[item],
            'class_names': class_names
        }
        image_transformed = self.transform(image.pixel_array.copy().astype('float32'))
        return image_transformed, target


    def __len__(self):
        return len(self.available_files)

    def get_true_df(self):
        if self.mode != 'test':
            return self.data_desc.loc[self.data_desc.image_id.isin(self.available_files)]




class XRAYShelveLoad:
    def __init__(
        self,
        mode = 'train',
        data_dir = '../data/chest_xray/',
        database_dir = '../data/chest_xray/',
        split = 0.8
    ):
        if mode not in ['train', 'test', 'eval']:
            raise KeyError('Mode needs to be in [train, test, eval]')
        self.transform = xray.utils.get_augmentation(prob= 0.8 if mode == 'train' else 0)

        self.available_files = [
            f.split('.')[0] for f in os.listdir(
                os.path.join(data_dir, 'test' if mode == 'test' else 'train')
            ) if f.endswith('dicom')
        ]
        if mode in ['train', 'eval']:
            self.database = shelve.open(
                os.path.join(database_dir, 'train_data.db'), flag='r', writeback=False
            )
            if mode == 'train':
                self.available_files = self.available_files[: int(len(self.available_files) * split)]
            else:
                self.available_files = self.available_files[int(len(self.available_files) * split):]

        else:
            self.database = shelve.open(
                os.path.join(database_dir, 'test_data.db'), flag='r', writeback=False
            )
        self.length = len(self.available_files)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        item_data = self.database[self.available_files[item]]

        image_transformed = self.transform(
            image=(np.expand_dims(item_data['image'], axis=2)),
            bboxes=item_data['bboxes'],
            class_labels=item_data['class_labels'],
            rad_id=item_data['rad_id'],
            image_name=self.available_files[item]
        )
        image_transformed['image'] = np.transpose(image_transformed['image'], axes=(2,0,1))/255
        image_transformed['image'] = torch.from_numpy(image_transformed['image']).float()

        labels = torch.Tensor([box[4] for box in image_transformed['bboxes']]).long()
        if labels.size()[0] == 0:
            labels = torch.tensor([14], dtype=torch.long)

        boxes = torch.Tensor([box[:4] for box in image_transformed['bboxes']]).float()
        if boxes.size()[0] == 0:
            boxes = torch.Tensor([[0,0,1,1]])

        boxes, labels = xray.utils.filter_radiologist_findings(boxes, labels)
        if len(labels) == 0:
            # TODO: do something more clever. This happens when radiologist cant decide on either
            #  class in the image
            boxes = torch.Tensor([[0,0,1,1]])
            labels = torch.tensor([14], dtype=torch.long)
        else:
            for i, label in enumerate(labels):
                if label.item() == 14:
                    boxes[i] = torch.Tensor([0,0,1,1])
        return image_transformed['image'], {
            'boxes': boxes,
            'labels': labels,
            'file_name': image_transformed['image_name']
        }


class VinBigDataset:
    def __init__(
        self,
        mode = 'train',
        data_dir = '../data/',
        split = 0.8
    ):
        if mode not in ['train', 'test', 'eval']:
            raise KeyError('Mode needs to be in [train, test, eval]')
        self.transform = xray.utils.get_augmentation(prob= 0.6 if mode == 'train' else 0)
        self.data_directory = os.path.join(data_dir, 'test' if mode == 'test' else 'train')
        self.available_files = [
            f.split('.')[0] for f in os.listdir(self.data_directory) if f.endswith('png')
        ]
        self.mode = mode

        if mode == 'train':
            self.available_files = self.available_files[: int(len(self.available_files) * split)]
        elif mode == 'eval':
            self.available_files = self.available_files[int(len(self.available_files) * split):]

        self.length = len(self.available_files)
        if self.mode != 'test':
            self.data_desc = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.data_directory, self.available_files[item]) + '.png')
        image_array = np.array(image)
        if self.mode != 'test':
            image_data_desc = self.data_desc.loc[
                self.data_desc['image_id'] == self.available_files[item]
            ]
            class_labels = image_data_desc.class_id.values
            bboxes = []
            for _, row in image_data_desc.iterrows():
                bbox = (row.x_min, row.y_min, row.x_max, row.y_max, row.class_id)
                if np.isnan(bbox[0]):
                    bboxes.append([0, 0, 1, 1, 14])
                else:
                    bbox_rescaled = xray.utils.resize_bbox(bbox[:4], (row.width, row.height), image_array.shape)
                    bboxes.append(bbox_rescaled + [bbox[4]])

            rad_id = image_data_desc.rad_id
        else:
            bboxes = []
            class_labels = []
            rad_id = []
        image_transformed = self.transform(
            image=np.stack([image_array, image_array, image_array], axis=2),
            bboxes=bboxes,
            class_labels=class_labels,
            rad_id=rad_id,
            image_name=self.available_files[item]
        )
        image_transformed['image'] = torch.tensor(image_transformed['image'], dtype=torch.float).permute(2,0,1)/255

        labels = torch.Tensor([box[4] for box in image_transformed['bboxes']]).long()
        if labels.size()[0] == 0:
            labels = torch.tensor([14], dtype=torch.long)

        boxes = torch.Tensor([box[:4] for box in image_transformed['bboxes']]).float()
        if boxes.size()[0] == 0:
            boxes = torch.Tensor([[0,0,1,1]])

        boxes, labels = xray.utils.filter_radiologist_findings(boxes, labels)
        if len(labels) == 0:
            # TODO: do something more clever. This happens when radiologist cant decide on either
            #  class in the image
            boxes = torch.Tensor([[0,0,1,1]])
            labels = torch.tensor([14], dtype=torch.long)
        else:
            for i, label in enumerate(labels):
                if label.item() == 14:
                    boxes[i] = torch.Tensor([0,0,1,1])
        return image_transformed['image'].float(), {
            'boxes': boxes,
            'labels': labels,
            'file_name': image_transformed['image_name']
        }

class ZeroToOneTransform():
    def __call__(self, image):
        return (image - image.min())/(image - image.min()).max()


def resize_bbox(bbox, shape_original, shape_new):
    """ bbox in format x_min, x_max, y_min, y_max"""

    x_min_new = int(bbox[0] * (shape_new[0]/shape_original[0]))
    y_min_new = int(bbox[1] * (shape_new[1]/shape_original[1]))
    x_max_new = int(bbox[2] * (shape_new[0]/shape_original[0]))
    y_max_new = int(bbox[3] * (shape_new[1]/shape_original[1]))
    return x_min_new, y_min_new, x_max_new, y_max_new