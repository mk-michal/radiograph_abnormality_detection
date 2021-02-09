import datetime

import pandas as pd


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def my_custom_collate(x):
    x = [(a,b) for a,b in x]
    return list(zip(*x))


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


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.datetime.today().strftime(fmt)