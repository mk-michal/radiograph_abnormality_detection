import os
import shelve
from functools import lru_cache
from typing import Dict, Optional, List

import numpy as np
import torch
from matplotlib import pyplot as plt, patches as patches


def plot_target_vs_true(image_array, results_bboxes, results_labels, target_bboxes, target_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Display the image
    ax1.imshow(image_array, cmap=plt.cm.bone)
    ax2.imshow(image_array, cmap=plt.cm.bone)

    # Create a Rectangle patch
    for bbox, label in zip(results_bboxes, results_labels):
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.text(bbox[0],bbox[1], label)

    for bbox, label in zip(target_bboxes, target_labels):
        rect = patches.Rectangle(
            (bbox[0].item(), bbox[1].item()),
            bbox[2].item() - bbox[0].item(),
            bbox[3].item() - bbox[1].item(),
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax2.add_patch(rect)
        ax2.text(bbox[0].item(),bbox[1].item(), label.item())


    ax1.plot()
    ax2.plot()


def get_results_plot(
    results: Dict[str, np.array],
    image_name: str,
    targets: Optional[Dict[str, torch.Tensor]],
    database_set: str = 'train',
    database_dir: str = '../data/chest_xray/'
):
    db = get_database(database_set, database_dir)
    image_array = db[image_name]['image']
    if targets is None:
        plot_image_with_bboxes(image_array, results['boxes'], results['labels'])
    else:
        plot_target_vs_true(
            image_array, results['boxes'], results['labels'], targets['boxes'], targets['labels']
        )


def plot_image_with_bboxes(image: np.array, bboxes: List[np.array], labels: List[np.array]):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image, cmap=plt.cm.bone)

    # Create a Rectangle patch
    for bbox, label in zip(bboxes, labels):
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(bbox[0],bbox[1], label)

    plt.show()


@lru_cache
def get_database(databaset: str = 'train', database_dir: str = '../data/chest_xray/'):
    assert databaset in ['train', 'test']
    db = shelve.open(
        os.path.join(database_dir, 'train_data.db'), flag='r', writeback=False
    )
    return db