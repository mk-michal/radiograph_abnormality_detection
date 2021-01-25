import argparse
import datetime
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

parser = argparse.ArgumentParser()
parser.add_argument('--lr')

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.datetime.today().strftime(fmt)


def my_custom_collate(x):
    x = [(a,b) for a,b in x if b['boxes'].size()[0] != 0]
    return list(zip(*x))

def create_true_df(descriptions):
    true_df = pd.DataFrame(
        columns=['image_id', 'class_name', 'class_id', 'x_min', 'y_min', 'x_max', 'y_max']
    )


    for desc in descriptions:
        bbox_df_true = pd.DataFrame(
            desc['boxes'].numpy(), columns=['x_min', 'y_min', 'x_max', 'y_max']
        )

        bbox_df_true['class_id'] = desc['labels']
        bbox_df_true['image_id'] = desc['file_name']

        true_df = true_df.append(bbox_df_true)

    return true_df



def create_eval_df(results, descriptions):
    eval_df = pd.DataFrame(columns=['image_id','PredictionString'])

    for result, desc in zip(results, descriptions):
        string_bboxes = list(map(
            lambda x: ' '.join([str(i.item()) for i in  x.long()]) if len(x.size()) !=0 else '14 1 0 0 1 1',
            result['boxes']
        ))
        string_scores = []
        for bbox, score, pred_class in zip(string_bboxes, result['scores'], result['labels']):
            string_scores.append(f'{pred_class.item()} {score.item()} {bbox}')


    return eval_df




    pass


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
            all_results = []
            all_targets = []

            for i, (x_eval, x_target) in enumerate(eval_loader):
                results = model(x_eval)
                # evaluate_result(all_results)
                target_values.extend(x_target)

                all_results.extend(results)
                all_targets.extend(x_target)
                if i >= 2:
                    break

            true_df = create_true_df(descriptions=x_target)
            eval_df = create_eval_df(results=all_results,descriptions=x_target)
            vinbigeval = VinBigDataEval(true_df)
            final_evaluation = vinbigeval.evaluate(eval_df)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train()



