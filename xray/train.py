import argparse
import logging
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torch.optim import SGD

from xray.evaluation import VinBigDataEval
from xray.data_preprocessing import XRayDataset
from xray.utils import time_str

parser = argparse.ArgumentParser()
parser.add_argument('--lr')


def my_custom_collate(x):
    x = [(a,b) for a,b in x if b['boxes'].size()[0] != 0]
    return list(zip(*x))

def create_final_df(results, descriptions):


def train():

    logger = logging.getLogger('Training')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 15)
    model.to('cpu')

    optimizer = SGD(model.parameters(), weight_decay=0.005, lr = 0.001, momentum=0.9)
    # optimizer.to('cpu')

    train_loader = DataLoader(
        XRayDataset('train'),
        shuffle=False,
        num_workers=0,
        batch_size=32,
        collate_fn=my_custom_collate
    )

    eval_loader = DataLoader(
        XRayDataset('eval'),
        shuffle=False,
        num_workers=0,
        batch_size=32,
        collate_fn=my_custom_collate
    )

    test_loader = DataLoader(
        XRayDataset('test'),
        shuffle=False,
        num_workers=0,
        batch_size=32,
    )

    logger.info('Starting training')
    for epoch in range(10):
        model.train()
        epoch_time = time.time()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            batch_time = time.time()
            loss_dict = model(x_batch, y_batch)
            total_loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()

            total_loss.backward()
            optimizer.step()
            logger.warning(
                f'{time_str()}, Step {step}/{len(train_loader)} in Ep {epoch}, {time.time() - batch_time:.2f}s '
                f'train_loss:{total_loss.item():.4f}'
            )
            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                logger.info(
                    f'{time_str()}, Step {step}/{len(train_loader)} in Ep {epoch}, {time.time() - batch_time:.2f}s '
                    f'train_loss:{total_loss.item():.4f}'
                )
                break


        logger.info(f'Epoch duration: {time.time() - epoch_time}')
        logger.info('==========================================')
        logger.info(f'Testing results after epoch {epoch + 1} on eval_loader {epoch + 1}')
        target_values = []

        with torch.no_grad():
            model.eval()
            all_results = torch.Tensor()
            for x_eval, x_target in eval_loader:
                results = model(x_eval)
                # evaluate_result(all_results)
                target_values.extend(x_target)
                torch.cat(all_results, results, dim = 0)


            final_df = create_final_df(target_values, results)
            eval_dataset = VinBigDataEval(true_df)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train()



