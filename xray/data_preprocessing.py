import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pydicom
import csv
from zipfile import ZipFile
import torchvision
import pandas as pd


class XRayDataset:
    def __init__(self, mode: str = 'train', data_dir: str = '../data/chest_xray/'):
        self.mode_dir = os.path.join(data_dir, mode)

        self.available_files = [
            f.split('.')[0] for f in os.listdir(self.mode_dir) if f.endswith('dicom')
        ]

        self.data_desc = pd.read_csv(os.path.join(data_dir, f'{mode}.csv'))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((400,400)),
            torchvision.transforms.ToTensor()
        ])


    def __getitem__(self, item):
        image = pydicom.dcmread(os.path.join(self.mode_dir, self.available_files[item] + '.dicom'))
        file_description = self.data_desc.loc[self.data_desc.image_id == self.available_files[item]]

        # TODO: Do IoU for the bboxes and make some mean for shared boxes > than lets say 0.4
        bboxes = [(row.x_min, row.x_max, row.y_min, row.y_max) for _, row in file_description.iterrows() if row.max is not None]
        labels = [row.image_id for _, row in file_description.iterrows()]

        target = {'bboxes': bboxes, 'labels': labels, 'name': self.available_files[item]}

        image_transformed = self.transform(image)
        return image_transformed, target


    def __len__(self):
        return len(self.available_files)

data = XRayDataset()
for x, target in data:
    print(x)
    print(target)

