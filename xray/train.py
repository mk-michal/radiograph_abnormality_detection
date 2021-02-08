import argparse
import copy
import datetime
import json
import logging
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torch.optim import SGD

from xray.evaluation import VinBigDataEval
from xray.dataset import XRAYShelveLoad

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', default='../data/chest_xray/', type=str)
parser.add_argument('--database-path', default='../data/chest_xray/', type=str)
parser.add_argument('--save-path', default='../data/chest_xray', type=str)
parser.add_argument('--n-workers', default=1, type=int)
parser.add_argument('-lr', default=0.01, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--log-step', default=20, type=int)



def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.datetime.today().strftime(fmt)


def my_custom_collate(x):
    x = [(a,b) for a,b in x]
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

    image_ids = []
    string_scores = []
    for result, desc in zip(results, descriptions):
        string_bboxes = list(map(
            lambda x: ' '.join([str(i.item()) for i in  x.long()]) if len(x.size()) !=0 else '14 1 0 0 1 1',
            result['boxes']
        ))
        for bbox, score, pred_class in zip(string_bboxes, result['scores'], result['labels']):
            image_ids.append(desc['file_name'])
            string_scores.append(f'{pred_class.item()} {score.item()} {bbox}')

    eval_df = pd.DataFrame({'image_id': image_ids, 'PredictionString': string_scores})

    return eval_df


def train():
    cfg = parser.parse_args()
    logger = logging.getLogger('Training')
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    model_path_folder = os.path.join(cfg.save_path, time_str())
    os.makedirs(model_path_folder, exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(model_path_folder, 'model_log.log'))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        trainable_backbone_layers=2,
        # num_classes=15,
        min_size=400,
        max_size=400,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 15)
    model.to(cfg.device)

    optimizer = SGD(model.parameters(), weight_decay=0.005, lr=cfg.lr, momentum=cfg.momentum)

    train_loader = DataLoader(
        XRAYShelveLoad('train', data_dir=cfg.data_path, database_dir=cfg.database_path),
        shuffle=True,
        num_workers=cfg.n_workers,
        batch_size=cfg.batch_size,
        collate_fn=my_custom_collate
    )

    eval_loader = DataLoader(
        XRAYShelveLoad('eval', data_dir=cfg.data_path, database_dir=cfg.database_path),
        shuffle=False,
        num_workers=cfg.n_workers,
        batch_size=cfg.batch_size,
        collate_fn=my_custom_collate
    )

    test_loader = DataLoader(
        XRAYShelveLoad('test', data_dir=cfg.data_path, database_dir=cfg.database_path),
        shuffle=False,
        num_workers=cfg.n_workers,
        batch_size=cfg.batch_size,
    )

    logger.info('Starting training')
    best_eval_ma = 0
    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_time = time.time()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = [x.to(cfg.device) for x in x_batch]
            y_batch = [
                {'boxes': j['boxes'].to(cfg.device), 'labels': j['labels'].to(cfg.device)} for j in y_batch
            ]
            # y_batch = y_batch.to(cfg.device)
            batch_time = time.time()
            loss_dict = model(x_batch, y_batch)
            total_loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()

            total_loss.backward()
            if (step + 1) % cfg.log_step == 0 or (step + 1) == len(train_loader):
                logger.info(
                    f'{time_str()}, Step {step}/{len(train_loader)} in Ep {epoch}, {time.time() - batch_time:.2f}s '
                    f'train_loss:{total_loss.item():.4f}'
                )


        logger.info(f'Epoch duration: {time.time() - epoch_time}')
        logger.info('==========================================')
        logger.info(f'Testing results after epoch {epoch + 1} on eval_loader {epoch + 1}')
        target_values = []

        with torch.no_grad():
            model.eval()
            all_results = []
            all_targets = []

            for i, (x_eval, x_target) in enumerate(eval_loader):
                x_eval = torch.stack(x_eval).to(cfg.device)
                results = model(x_eval)
                target_values.extend(x_target)

                all_results.extend(results)
                all_targets.extend(x_target)

            true_df = create_true_df(descriptions=all_targets)
            eval_df = create_eval_df(results=all_results,descriptions=all_targets)
            vinbigeval = VinBigDataEval(true_df)
            final_evaluation = vinbigeval.evaluate(eval_df)
            if final_evaluation.stats[0] > best_eval_ma:
                best_model = copy.deepcopy(model)

    logger.info("===================================================================")
    logger.info("Testing best model on test set")
    with torch.no_grad():
        best_model.eval()
        all_results = []
        all_targets = []

        for i, (x_test, x_target) in enumerate(test_loader):
            x_test = torch.stack(x_test).to(cfg.device)
            # x_target['bboxes'] = x_target['bboxes'].to(cfg.device)

            results = best_model(x_test)
            target_values.extend(x_target)

            all_results.extend(results)
            all_targets.extend(x_target)

    final_test_df = create_eval_df(x_test)

    best_model_path = os.path.join(model_path_folder, 'best_model.cfg')
    logger.info(f'Saving best model to {best_model_path}')
    torch.save(best_model.state_dict(), best_model_path)
    with open(os.path.join(model_path_folder, 'model_hyperparameters.json'), 'w') as j:
        json.dump(cfg.__dict__, j)

    final_results_save_path = os.path.join(model_path_folder, 'final_result_test.csv')
    logger.info(f'Saving final results for tests set into {final_results_save_path}  ')
    final_test_df.to_csv(final_results_save_path)



if __name__ == '__main__':
    train()



