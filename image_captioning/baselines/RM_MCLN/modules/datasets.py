import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.aug = args.aug

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(
                self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir,
                                          image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir,
                                          image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        if self.aug == 'original':
            image = Image.open(os.path.join(self.image_dir,
                                            image_path[0])).convert('RGB')
        elif self.aug == 'ours':
            image = cv2_imread(os.path.join(self.image_dir, image_path[0]))
        else:
            raise NotImplementedError()
        if self.transform is not None:
            if self.aug == 'original':
                image = self.transform(image)
            elif self.aug == 'ours':
                res = self.transform(image=image)
                image = res['image']
            else:
                raise NotImplementedError()
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


def cv2_imread(path):
    img = cv2.imread(path)
    assert img is not None, f'cannot read {path}'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
