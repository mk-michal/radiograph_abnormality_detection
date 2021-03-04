import os
import unittest

import torch

from xray.utils import filter_radiologist_findings

class TrainPipelin(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_filter_radiologists(self):

        boxes = torch.tensor([
            [868.8510, 242.3215, 900.4315, 395.5031],
            [263.5402, 315.9097, 362.8567, 428.8611],
            [698.1447, 493.1744, 801.4595, 609.5676],
            [265.7943, 255.7272, 439.2505, 444.5807],
            [343.9662, 261.4236, 428.3037, 383.6049],
            [343.9662, 261.4236, 428.3037, 383.6049],
            [242.6154, 304.7392, 351.8375, 440.0254],
            [263.5402, 315.9097, 362.8567, 428.8611],
            [697.1730, 492.4722, 792.9791, 611.1887],
            [330.0433, 262.3767, 426.3797, 411.6760]
        ])

        labels = torch.tensor([12,  8,  1,  9,  9,  8,  9,  9,  1,  9])
        boxes, labels = filter_radiologist_findings(boxes, labels, iou_threshold=0.5)
        assert len(labels) == 3
        assert labels[0] == 1









