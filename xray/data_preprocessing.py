# main bulk of the processing code taken from https://www.kaggle.com/bjoernholzhauer/eda-dicom-reading-vinbigdata-chest-x-ray

import argparse
import functools
import os
from pydicom.pixel_data_handlers.util import apply_voi_lut
import albumentations
import pydicom
import warnings

from fastcore.parallel import parallel
import shelve
import re
import pandas as pd
import numpy as np

# Using function from another great notebook: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
def read_xray(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if voi_lut:

            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    data = np.stack([data] * 3).transpose(1, 2, 0)

    return data


# This function will read a .dicom file, turn the smallest side to 600 pixels, and then save the additional annotations together with the image into a dictionary
def get_and_save(x, directory: str, mode: str = 'train'):
    idx = x[0]
    image_id = x[1]

    transform = albumentations.Compose(
        [albumentations.SmallestMaxSize(max_size=600, always_apply=True)],
        bbox_params=albumentations.BboxParams(format='pascal_voc')
    )
    img = read_xray(path=os.path.join(directory, mode, image_id + '.dicom'))
    if mode == 'train':
        train = pd.read_csv(os.path.join(directory, 'train.csv'))
        rad_id = np.array([int(re.findall(r'\d+', rad_id)[0]) for rad_id in
                           train.loc[train['image_id'] == image_id, 'rad_id'].values], dtype=np.int8)
        class_labels = train.loc[train['image_id'] == image_id, 'class_id'].values
        bboxes = [list(row) for rowid, row in train.loc[
            train['image_id'] == image_id, ['x_min', 'y_min', 'x_max', 'y_max', 'class_id']].fillna(
            {'x_min': 0, 'y_min': 0, 'x_max': 1, 'y_max': 1}).astype(np.int16).iterrows()]
        transformed = transform(image=img,
                                bboxes=bboxes,
                                class_labels=class_labels)
        return dict(image_id=image_id,
                    image=transformed['image'][:, :, 0],
                    rad_id=rad_id,
                    bboxes=np.array(transformed['bboxes'], dtype=np.float32),
                    class_labels=transformed['class_labels'].astype(np.int8))
    elif mode == 'test':
        transformed = transform(
            image=img,
            bboxes=[],
            class_labels=np.array([])
        )
        return dict(image_id=image_id,
                    image=transformed['image'][:, :, 0],
                    rad_id=np.array([]),
                    bboxes=np.array(transformed['bboxes'], dtype=np.float32),
                    class_labels=transformed['class_labels'].astype(np.int8))
    else:
        KeyError(f'Mode needs to be one of [train, test], {mode} was given ')



if __name__ == '__main__':
    # Parallel processing of the .dicom files
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-workers', default=8, type=int)
    parser.add_argument('--data-path', default='../data/chest_xray/')
    parser.add_argument('--data-path-output', default='../data/chest_xray/')

    parser.add_argument('--mode', default='train')
    cfg = parser.parse_args()

    data_dir = os.path.join(cfg.data_path, cfg.mode)
    list_of_images = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('dicom')]

    out1 = parallel(
        functools.partial(get_and_save, directory=cfg.data_path, mode=cfg.mode ),
        [(idx, image_id) for idx, image_id in enumerate(list_of_images)],
        n_workers=cfg.n_workers,
        progress=True
    )

    with shelve.open(os.path.join(cfg.data_path_output, f'{cfg.mode}_data.db')) as myshelf:
        myshelf.update( { dictentry['image_id']: {'image': dictentry['image'],
                                                  'rad_id': dictentry['rad_id'],
                                                  'bboxes': dictentry['bboxes'],
                                                  'class_labels': dictentry['class_labels'] }  for dictentry in out1 } )


