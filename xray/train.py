import argparse
import copy
import json
import logging
import os
import time
import traceback

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torch.optim import SGD

import xray.dataset
import xray.evalutation
import xray.utils

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', default='../data/chest_xray/vinbigdata/', type=str)
parser.add_argument('--database-path', default='../data/chest_xray/vinbigdata/', type=str)
parser.add_argument('--save-path', default='../data/chest_xray', type=str)
parser.add_argument('--checkpoint-path', default=None)
parser.add_argument('--n-workers', default=1, type=int)
parser.add_argument('-lr', default=0.01, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--log-step', default=20, type=int)
parser.add_argument('--last-epoch', default=-1, type=int)
parser.add_argument('--gamma', default=0.02, type=float)
parser.add_argument('--step-size', default=10, type=int)
parser.add_argument('--weight-decay', default=0.005, type=float)



def train(model_path_folder, cfg, logger):
    if cfg.checkpoint_path:
        model = xray.evalutation.get_rcnn(cfg.checkpoint_path)
        model.to(cfg.device)


    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            trainable_backbone_layers=2,
            # num_classes=15,
            min_size=1024,
            max_size=1024,
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 15)
        model.to(cfg.device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = SGD(params, weight_decay=cfg.weight_decay, lr=cfg.lr, momentum=cfg.momentum)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, gamma=cfg.gamma, step_size=cfg.step_size, last_epoch=cfg.last_epoch
    )

    train_loader = DataLoader(
        xray.dataset.VinBigDataset('train', data_dir=cfg.data_path),
        shuffle=True,
        num_workers=cfg.n_workers,
        batch_size=cfg.batch_size,
        collate_fn=xray.utils.my_custom_collate,
        pin_memory=True
    )

    eval_loader = DataLoader(
        xray.dataset.VinBigDataset('eval', data_dir=cfg.data_path),
        shuffle=False,
        num_workers=cfg.n_workers,
        batch_size=cfg.batch_size,
        collate_fn=xray.utils.my_custom_collate,
        pin_memory=True
    )

    logger.info('Starting training')
    best_eval_ma = 0

    average_loss = xray.utils.Averager()
    try:
        for epoch in range(cfg.n_epochs):
            average_loss.reset()
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
                optimizer.step()
                average_loss.send(total_loss.item())
                if (step + 1) % cfg.log_step == 0 or (step + 1) == len(train_loader):
                    logger.info(
                        f'{xray.utils.time_str()}, Step {step}/{len(train_loader)} in Ep {epoch}, {time.time() - batch_time:.2f}s '
                        f'train_loss:{average_loss.value:.4f}'
                    )
            lr_scheduler.step()


            logger.info(f'Epoch duration: {(time.time() - epoch_time)/60} Min')
            logger.info('==========================================')
            logger.info(f'Testing results after epoch {epoch + 1} on eval_loader {epoch + 1}')

            all_results, all_targets = xray.evalutation.model_eval_forward(model, eval_loader, cfg.device)
            final_evaluation = xray.evalutation.calculate_metrics(all_results, all_targets)
            logger.info(f'Ma metric on evaluation dataset after epoch {epoch} is with '
                        f'IoU 0.4 is {final_evaluation.stats[0]}')

            if final_evaluation.stats[0] > best_eval_ma:
                best_eval_ma = final_evaluation.stats[0]
                logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'*2)
                logger.info(f'New best model after epoch {epoch} with ma {final_evaluation.stats[0]}')
                logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'*2)
                best_model = copy.deepcopy(model)
                best_model_path = os.path.join(model_path_folder, 'best_model_rcnn.cfg')
                logger.info(f'Saving best model to {best_model_path}')
                torch.save(best_model.state_dict(), best_model_path)

        return best_model
    except Exception as e:
        logger.exception(f'Training failed with exception {e}')
        traceback.print_exc()
        return best_model


def create_test_submission(model, model_path_folder, cfg, logger):
    test_loader = DataLoader(
        xray.dataset.VinBigDataset('test', data_dir=cfg.data_path),
        shuffle=False,
        num_workers=cfg.n_workers,
        batch_size=cfg.batch_size,
        collate_fn=xray.utils.my_custom_collate,
        pin_memory=True
    )

    logger.info("===================================================================")
    logger.info("Testing best model on test set")
    all_results, all_targets = xray.evalutation.model_eval_forward(
        model, test_loader, cfg.device, score_threshold=0.5
    )
    logger.info("Creating submission file for test data ...")

    all_results = xray.utils.no_findings_to_ones(all_results)
    submission_file = xray.utils.create_submission_df(all_results, [i['file_name'] for i in all_targets])
    submission_file = xray.utils.rescale_to_original_size(submission_file)
    submission_file['PredictionString'] = submission_file.PredictionString.apply(xray.utils.do_nms)

    with open(os.path.join(model_path_folder, 'model_hyperparameters.json'), 'w') as j:
        json.dump(cfg.__dict__, j)

    final_results_save_path = os.path.join(model_path_folder, 'final_submission.csv')
    logger.info(f'Saving final results for tests set into {final_results_save_path}  ')
    submission_file.to_csv(final_results_save_path, header=True, index=False)



if __name__ == '__main__':
    cfg = parser.parse_args()
    model_path_folder = os.path.join(cfg.save_path, xray.utils.time_str())
    os.makedirs(model_path_folder, exist_ok=True)

    logger = xray.utils.define_logger('Train pipeline', folder=model_path_folder)

    model = train(model_path_folder, cfg, logger)
    create_test_submission(model, model_path_folder, cfg, logger)




