import logging
import os
import shelve

import xray.utils

import albumentations.pytorch
import albumentations as A
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
        self.transform = xray.utils.get_augmentation(prob= 0.6 if mode == 'train' else 0)

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
            labels = torch.Tensor([14]).long()

        boxes = torch.Tensor([box[:4] for box in image_transformed['bboxes']]).float()
        if boxes.size()[0] == 0:
            boxes = torch.Tensor([[0,0,1,1]])
        return image_transformed['image'], {
            'boxes': boxes,
            'labels': labels,
            'file_name': image_transformed['image_name']
        }
        # item_data['image'] = np.expand_dims(item_data['image'], axis=2)/255
        # item_data['image'] = np.transpose(item_data['image'], axes=(2, 0, 1))
        # item_data['image'] = torch.from_numpy(item_data['image']).float()
        # return item_data['image'], {
        #     'boxes': torch.Tensor([box[:4] for box in item_data['bboxes']]),
        #     'labels': torch.Tensor(item_data['class_labels']).long(),
        #     'file_name': self.available_files[item]
        # }


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