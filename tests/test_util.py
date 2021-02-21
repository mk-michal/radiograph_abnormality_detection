import unittest

from xray.dataset import XRAYShelveLoad
from xray.utils import filter_radiologist_findings

class TrainPipelin(unittest.TestCase):
    def setUp(self) -> None:

        self.dataset = XRAYShelveLoad(
            'eval', data_dir='../data/chest_xray/',database_dir='../data/chest_xray/'
        )

    def test_filter_radiologists(self):
        _, result = self.dataset[21]
        boxes, labels = filter_radiologist_findings(result['boxes'], result['labels'])
        assert len(labels == 1)
        assert labels[0].item() == 14

        _, result = self.dataset[2]
        boxes, labels = filter_radiologist_findings(result['boxes'], result['labels'])
        assert len(labels == 1)
        assert labels[0].item() == 6

        _, result = self.dataset[3]
        boxes, labels = filter_radiologist_findings(result['boxes'], result['labels'])
        assert len(labels == 2)
        assert labels[0].item() == 13
        assert labels[1].item() == 11

