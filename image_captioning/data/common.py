from albumentations.pytorch import ToTensorV2
from trainer.start import *


class GrayToTensor(ToTensorV2):
    def apply(self, img, **params):
        # because of gray scale
        # we add an additional channel
        return torch.from_numpy(img).unsqueeze(0)


@dataclass
class BaseDataConfig(BaseConfig):
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True

    @property
    def name(self):
        raise NotImplementedError()

    def make_dataset(self):
        raise NotImplementedError()


class BaseDataset:
    def __init__(self, conf: BaseDataConfig):
        self.conf = conf

    def make_loader(self, data, shuffle):
        return DataLoader(
            data,
            batch_size=self.conf.batch_size,
            shuffle=shuffle,
            pin_memory=self.conf.pin_memory,
            num_workers=self.conf.num_workers,
            multiprocessing_context=(mp.get_context('fork')
                                     if self.conf.num_workers > 0 else None),
        )

    def evaluate_instance(self, pred, gt):
        raise NotImplementedError()

    def evaluate(self, pred, gt):
        raise NotImplementedError()

    @property
    def metric(self):
        return []
