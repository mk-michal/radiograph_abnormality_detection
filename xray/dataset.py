import logging
import os

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

        self.logger = logging.getLogger('XRayDataset')
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