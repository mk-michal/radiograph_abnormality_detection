import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pydicom
import csv
from zipfile import ZipFile

import torch
import torchvision
import pandas as pd
import numpy as np

def resize_bbox(bbox, shape_original, shape_new):
    """ bbox in format x_min, x_max, y_min, y_max"""

    x_min_new = int(bbox[0] * (shape_new[0]/shape_original[0]))
    y_min_new = int(bbox[1] * (shape_new[1]/shape_original[1]))
    x_max_new = int(bbox[2] * (shape_new[0]/shape_original[0]))
    y_max_new = int(bbox[3] * (shape_new[1]/shape_original[1]))
    return x_min_new, y_min_new, x_max_new, y_max_new


class ZeroToOneTransform():
    def __call__(self, image):
        return (image - image.min())/(image - image.min()).max()




class XRayDataset:
    def __init__(self, mode: str = 'train', data_dir: str = '../data/chest_xray/'):
        self.mode_dir = os.path.join(data_dir, mode)

        self.available_files = [
            f.split('.')[0] for f in os.listdir(self.mode_dir) if f.endswith('dicom')
        ]
        self.logger = logging.getLogger('XRayDataset')
        self.data_desc = pd.read_csv(os.path.join(data_dir, f'{mode}.csv'))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((400,400)),
            torchvision.transforms.ToTensor(),
            ZeroToOneTransform()
        ])


    def __getitem__(self, item, max_bboxes: int = 16):
        image = pydicom.read_file(os.path.join(self.mode_dir, self.available_files[item] + '.dicom'))
        file_description = self.data_desc.loc[self.data_desc.image_id == self.available_files[item]]

        # bboxes = torch.zeros(max_bboxes, 4)
        # labels = torch.zeros(max_bboxes)
        # TODO: Do IoU for the bboxes and make some mean for shared boxes > than lets say 0.4

        class_names, bboxes, labels = [], [], []
        for _, row in file_description.iterrows():
            if not np.isnan(row.x_max):

                bboxes.append((row.x_min, row.y_min, row.x_max, row.y_max))
                labels.append(row.class_id)
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
