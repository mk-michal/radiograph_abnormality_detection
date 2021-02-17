import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from tqdm import tqdm
from xray.coco_eval import VinBigDataEval
from xray.dataset import XRAYShelveLoad
from xray.utils import create_true_df, create_eval_df, my_custom_collate

best_model_path = '../data/chest_xray/2021-02-09_17:46:42/rcnn_checkpoint.pth'


def get_rcnn(model_path, device: str = 'cpu'):
    model = fasterrcnn_resnet50_fpn(pretrained_backbone=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 15)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    return model


def model_eval_forward(
    model: FasterRCNN, loader: DataLoader, device: str = 'cpu', score_threshold = 0.5
):
    with torch.no_grad():
        model.eval()
        all_results = []
        all_targets = []

        for i, (x_eval, x_target) in tqdm(enumerate(loader), total=len(loader)):
            x_eval = [x.to(device) for x in x_eval]
            results = model(x_eval)

            all_results.extend(results)
            all_targets.extend(x_target)

    index_selected = [results['scores'].data.cpu().numpy() > score_threshold for results in all_results]
    results_selected = [{r: val[index].data.cpu().numpy() for r, val in result.items()} for result, index in zip(all_results, index_selected)]

    return results_selected, all_targets


def calculate_metrics(results, targets):
    true_df = create_true_df(descriptions=targets)
    eval_df = create_eval_df(results=results, description=targets)
    vinbigeval = VinBigDataEval(true_df)
    final_evaluation = vinbigeval.evaluate(eval_df)
    return final_evaluation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default = '../data/xray-kaggle/best_model_rcnn.cfg')
    cfg = parser.parse_args()

    dataset = XRAYShelveLoad('train', data_dir='../data/chest_xray', database_dir='../data/chest_xray')
    dataset.length = 10
    eval_loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=0,
        batch_size=8,
        collate_fn=my_custom_collate
    )

    model = get_rcnn(cfg.model_path)
    results, targets = model_eval_forward(model, eval_loader)
    df = create_eval_df(results, targets)
    final_metric = calculate_metrics(results, targets)


