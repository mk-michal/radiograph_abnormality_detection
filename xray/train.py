from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torch.optim import SGD
from xray.data_preprocessing import XRayDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', )

def my_custom_collate(x):
    x = [(a,b) for a,b in x if b['boxes'].size()[0] != 0]
    return list(zip(*x))


def train():

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 14)
    model.to('cpu')

    optimizer = SGD(model.parameters(), weight_decay=0.005, lr = 0.005, momentum=0.9)
    # optimizer.to('cpu')

    dataset_loader = DataLoader(
        XRayDataset('train'),
        shuffle=False,
        num_workers=0,
        batch_size=6,
        collate_fn=my_custom_collate

    )
    for epoch in range(10):
        for x_batch, y_batch in dataset_loader:
            loss_dict = model(x_batch, y_batch)
            total_loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()

            total_loss.backward()
            optimizer.step()




if __name__ == '__main__':
    train()



