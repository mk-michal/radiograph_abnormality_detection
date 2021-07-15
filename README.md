# Radiograph Abnormality Detection

Project for localizing and detection 14 types of thoratic abnormalities from ches radiographs.

## Dataset
We used kaggle VinBig dataset that can be found via following link: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data

For faster training we recommend using reshaped data to 1024x1024 that are available via https://www.kaggle.com/awsaf49/vinbigdata-1024-image-dataset

Dataset consists of 18 000 scans in DiCOM format that were annotated by experienced radiologists with multiple annotations for each image. Radiologists localized and classified 14 differend findings:

```
0 - Aortic enlargement
1 - Atelectasis
2 - Calcification
3 - Cardiomegaly
4 - Consolidation
5 - ILD
6 - Infiltration
7 - Lung Opacity
8 - Nodule/Mass
9 - Other lesion
10 - Pleural effusion
11 - Pleural thickening
12 - Pneumothorax
13 - Pulmonary fibrosis
```
Example Image from dataset:

![image](https://user-images.githubusercontent.com/35899621/125815413-0dcbf0b2-8562-46b1-9d4f-34350effb2f1.png)

## Method
For localization and classification of lung abnormalities we used two-staged model Faster-RCNN from pytorch library. ResNet50 was used as a base backbone of our model. We used pretrained model on ImageNET dataset and fine-tuned it on our data.

SGD optimizer with weight decay and momentum to optimize parameters of our model with Stepwise learning rate decay, that was decrease by factor of 0.1 each 30 epochs.  

Model was evaluated on unlabeled test set and results were submitted for VinBig kaggle competition. 

## Usage
Install requirements listed in `requirements.txt` by `pip install -r requirements.txt`

In order to run the training script you can run 

```
python xray/train.py --n-workers 8 \
    --data-path $DATA_PATH \
    --save-path $SAVE_PATH \
    --lr 0.001 \
    --device cuda \
    --momentum 0.9 \ 
    --n-epochs 100 \
    --batch-size 32 \
    --log-step 20 \
    --step-size 30 \
    --gamma 0.1
```

with proper for VinBig data path and save path.








