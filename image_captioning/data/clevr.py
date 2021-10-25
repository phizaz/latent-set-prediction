from data.common import BaseDataConfig, BaseDataset
import pickle
from tokenizer import load_roberta_tokenizer
import cv2
from albumentations import *
from albumentations.pytorch import ToTensorV2
from transformers import PreTrainedTokenizerFast
from trainer.start import *

# imagenet's
MEAN = [0.485, 0.456, 0.406]
SD = [0.229, 0.224, 0.225]

here = os.path.join(os.path.dirname(__file__), 'clevr_text')

split_options = {
    'clevr': (f'{here}/clevr_train_train.csv', f'{here}/clevr_train_val.csv',
              f'{here}/clevr_val.csv'),
    'clean':
    (f'{here}/clevr_clean_train_train.csv',
     f'{here}/clevr_clean_train_val.csv', f'{here}/clevr_clean_val.csv'),
}

ATTRS = [
    'shape', 'color', 'material', 'size', 'left_right', 'front_back', 'cnt'
]
VARIANTS = {
    'shape': ['cube', 'cylinder', 'sphere'],
    'color':
    ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow'],
    'material': ['metal', 'rubber'],
    'size': ['large', 'small'],
    'left_right': ['left', 'right'],
    'front_back': ['front', 'back'],
    'cnt': [1, 2, 3],
}


class CLEVRMetricCb(CollectCb):
    def __init__(self, keys):
        # collect 'pred' and 'y'
        super().__init__(keys=keys)
        self.keys = keys

    def on_ep_end(self, buffer, i_itr, **kwargs):
        pred = buffer[self.keys[0]]
        y = buffer[self.keys[1]]
        metric = CLEVRCombinedDataset.evaluate(pred, y)
        info = {
            'i_itr': i_itr,
            **metric,
        }
        self.add_to_bar_and_hist(info)
        self._flush()


@dataclass
class Object:
    shape: str = None
    material: str = None
    size: str = None
    color: str = None
    left_right: str = None
    front_back: str = None
    cnt: int = 1

    def __hash__(self):
        return hash((self.shape, self.material, self.size, self.color,
                     self.left_right, self.front_back, self.cnt))

    @classmethod
    def parse_from_int_tuple(self, obj):
        args = {}
        for i, k in enumerate(VARIANTS.keys()):
            args[k] = VARIANTS[k][obj[i]]
        return Object(**args)

    def to_int_tuple(self):
        out = []
        for k in VARIANTS.keys():
            v = getattr(self, k)
            v = VARIANTS[k].index(v)
            out.append(v)
        return tuple(out)

    @classmethod
    def parse_from_json_dict(self, obj):
        x, y, _ = obj['pixel_coords']
        left_right = 'left' if x < 480 / 2 else 'right'
        front_back = 'front' if y > 320 / 2 else 'back'
        obj = Object(
            shape=obj['shape'],
            color=obj['color'],
            size=obj['size'],
            material=obj['material'],
            left_right=left_right,
            front_back=front_back,
        )
        return obj

    @classmethod
    def parse_from_dict(self, obj):
        obj = Object(
            shape=obj['shape'],
            color=obj['color'],
            size=obj['size'],
            material=obj['material'],
            left_right=obj['left_right'],
            front_back=obj['front_back'],
            cnt=obj['cnt'],
        )
        return obj

    @classmethod
    def parse_from_text(self, text):
        num = {
            'one': 1,  # doesn't really exist
            'two': 2,
            'three': 3,
        }
        sections = text.split(' ')
        # concatenate strings of the same kind
        args = defaultdict(lambda: '')
        for each in sections:
            for k, v in VARIANTS.items():
                if k == 'cnt':
                    # silly but works
                    try:
                        args[k] = num[each]
                    except KeyError:
                        pass
                else:
                    if each in v:
                        args[k] += each
        obj = Object(**args)
        return obj

    def is_subset(self, obj: 'Object'):
        if self.shape is not None and self.shape != obj.shape:
            return False
        if self.material is not None and self.material != obj.material:
            return False
        if self.size is not None and self.size != obj.size:
            return False
        if self.color is not None and self.color != obj.color:
            return False
        if self.left_right is not None and self.left_right != obj.left_right:
            return False
        if self.front_back is not None and self.front_back != obj.front_back:
            return False
        if self.cnt is not None and self.cnt != obj.cnt:
            return False
        return True

    def to_str(self):
        name = []
        if self.cnt == 2:
            name.append('two')
        elif self.cnt == 3:
            name.append('three')
        elif self.cnt > 3:
            raise NotImplementedError()
        if self.front_back is not None:
            name.append(self.front_back)
        if self.left_right is not None:
            name.append(self.left_right)
        if self.size is not None:
            name.append(self.size)
        if self.color is not None:
            name.append(self.color)
        if self.material is not None:
            name.append(self.material)
        name.append(self.shape)
        return ' '.join(name)


@dataclass
class CLEVRTransformConfig(BaseConfig):
    size: int = 256
    rotate: int = 0
    rotate_p: float = 0.5
    brightness: float = 0
    contrast: float = 0
    bc_prob: float = 0
    min_size: float = 1
    crop_p: float = 0
    interpolation: str = 'cubic'

    @property
    def name(self):
        name = f'{self.size}size'
        if self.crop_p > 0:
            name += f'-min{self.min_size}p{self.crop_p}'
        if self.rotate > 0:
            name += f'-rot{self.rotate}p{self.rotate_p}'
        if self.bc_prob > 0:
            name += f'-bc({self.brightness},{self.contrast})p{self.bc_prob}'
        if self.interpolation != 'cubic':
            name += f'-{self.interpolation}'
        return name


@dataclass
class CLEVRDataConfig(BaseDataConfig):
    tokenizer: PreTrainedTokenizerFast = load_roberta_tokenizer('clevr')
    split: str = 'clevr'
    type: str = 'sentences'
    sort_by: str = 'alphabetical'
    csv_dir: str = f'data'
    img_dir: str = f'{here}/images'
    do_pad: bool = False
    n_max_sentences: int = None
    trans_conf: CLEVRTransformConfig = None

    @property
    def name(self):
        name = f'bs{self.batch_size}_clevr2'
        if self.split != 'clevr':
            name += f'-{self.split}'
        name += f'-sort{self.sort_by}_{self.trans_conf.name}'
        if self.do_pad:
            name += f'-pad'
        return name

    def make_dataset(self):
        return CLEVRCombinedDataset(self)


class CLEVRCombinedDataset(BaseDataset):
    def __init__(self, conf: CLEVRDataConfig) -> None:
        self.conf = conf

        train, val, test = split_options[self.conf.split]
        if conf.trans_conf is None:
            train_transform = None
            eval_transform = None
        else:
            train_transform = make_transform('train', conf.trans_conf)
            eval_transform = make_transform('eval', conf.trans_conf)
        if conf.type == 'sentences':
            data_cls = CLEVRDataset
        elif conf.type == 'text':
            data_cls = CLEVRTextDataset
        else:
            raise NotImplementedError()
        self.train = data_cls(
            train,
            'train',
            conf,
            train_transform,
            return_index=True,
            index_offset=0,
            cache_path=f'{here}/cache/{conf.split}_train.pkl',
        )
        self.val = data_cls(
            val,
            'train',
            conf,
            eval_transform,
            return_index=True,
            index_offset=len(self.train),
            cache_path=f'{here}/cache/{conf.split}_val.pkl',
        )
        self.test = data_cls(
            test,
            'val',
            conf,
            eval_transform,
            return_index=False,
            cache_path=f'{here}/cache/{conf.split}_test.pkl',
        )

    def make_loader(self, data, shuffle):
        if self.conf.type == 'sentences':
            collate_fn = SentencesCollator(self.conf.tokenizer)
        elif self.conf.type == 'text':
            collate_fn = TextCollator(self.conf.tokenizer)
        else:
            raise NotImplementedError()
        return DataLoader(
            data,
            batch_size=self.conf.batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            pin_memory=self.conf.pin_memory,
            num_workers=self.conf.num_workers,
            multiprocessing_context=(mp.get_context('fork')
                                     if self.conf.num_workers > 0 else None),
        )

    @property
    def metric(self):
        return [
            'micro_precision', 'micro_recall', 'micro_f1', 'macro_precision',
            'macro_recall', 'macro_f1'
        ]

    @classmethod
    def evaluate_instance(cls, pred_texts: List[str], gt: List[dict]):
        pred_objs = [
            Object.parse_from_text(text) for text in pred_texts
            if text.strip() != ''
        ]
        gt = [Object.parse_from_dict(each) for each in gt]
        pred_matches = [False] * len(pred_objs)
        gt_matches = [False] * len(gt)
        for i, pred in enumerate(pred_objs):
            is_subset = []
            gt_idx = None
            for j, tgt in enumerate(gt):
                is_subset.append(pred.is_subset(tgt))
                if is_subset[j]:
                    gt_idx = j
            # matches only one gt
            if sum(is_subset) == 1:
                gt_matches[gt_idx] = True
                pred_matches[i] = True

        return {
            'precision': sum(pred_matches) / (len(pred_objs) + 1e-8),
            'recall': sum(gt_matches) / (len(gt) + 1e-8),
        }

    @classmethod
    def evaluate(cls,
                 pred_texts: List[List[str]],
                 gt: List[List[dict]],
                 prefix: str = ''):
        """
        Args:
            pred_texts: List of List of strings
            gt: List of List of objects (in dict form)
            ignore_duplicates: resolving duplicated predictions automatically
        """
        assert len(pred_texts) == len(gt)

        n = len(pred_texts)
        precs = np.zeros(n)
        recs = np.zeros(n)
        pred_cnt = np.zeros(n)
        gt_cnt = np.zeros(n)
        for i in range(len(pred_texts)):
            score = cls.evaluate_instance(pred_texts[i], gt[i])
            n_pred = len(pred_texts[i])
            n_gt = len(gt[i])
            precs[i] = score['precision']
            recs[i] = score['recall']
            pred_cnt[i] = n_pred
            gt_cnt[i] = n_gt
        micro_prec = (precs * pred_cnt).sum() / pred_cnt.sum()
        micro_rec = (recs * gt_cnt).sum() / gt_cnt.sum()
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        macro_prec = precs.mean()
        macro_rec = recs.mean()
        # wrong!
        macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec)
        return {
            f'{prefix}micro_precision': micro_prec,
            f'{prefix}micro_recall': micro_rec,
            f'{prefix}micro_f1': micro_f1,
            f'{prefix}macro_precision': macro_prec,
            f'{prefix}macro_recall': macro_rec,
            f'{prefix}macro_f1': macro_f1,
        }


class CLEVRDataset(Dataset):
    def __init__(
        self,
        csv: str,
        img_split: str,
        conf: CLEVRDataConfig,
        transform=None,
        return_index=True,
        index_offset=0,
        cache_path: str = None,
    ):
        self.conf = conf
        self.img_split = img_split
        self.df = pd.read_csv(csv)

        try:
            # load cache
            with open(cache_path, 'rb') as f:
                self.filename_idxs = pickle.load(f)
                print(f'loaded from cache {cache_path}')
        except Exception:
            # real computation
            self.filename_idxs = defaultdict(list)
            for i in tqdm(range(len(self.df))):
                filename = self.df.loc[i, 'image_filename']
                self.filename_idxs[filename].append(i)
            # save cache
            if cache_path is not None:
                if not os.path.exists(os.path.dirname(cache_path)):
                    os.makedirs(os.path.dirname(cache_path))
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.filename_idxs, f)

        self.filenames = list(self.filename_idxs.keys())
        self.transform = transform
        self.return_index = return_index
        self.index_offset = index_offset

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        ############
        # REPORT
        filename = self.filenames[idx]
        group = self.df.loc[self.filename_idxs[filename]]

        lines = list(group['text'])
        # select only shortest ones
        if self.conf.sort_by is None:
            both = enumerate(lines)
        elif self.conf.sort_by == 'shortest':
            both = sorted(enumerate(lines), key=lambda x: len(x[1]))
        elif self.conf.sort_by == 'alphabetical':
            both = sorted(enumerate(lines), key=lambda x: x[1])
        else:
            raise NotImplementedError()
        sort_i, lines = [], []
        for i, l in both:
            sort_i.append(i)
            lines.append(l)
        lines_gt = list(group['text_gt'].iloc[sort_i])

        # pad with empty lines
        if self.conf.do_pad:
            lines += [''] * (self.conf.n_max_sentences - len(lines))
            lines_gt += [''] * (self.conf.n_max_sentences - len(lines_gt))

        res = self.conf.tokenizer(lines,
                                  return_attention_mask=False,
                                  add_special_tokens=False)
        input_ids = res['input_ids']
        # add bos
        for i in range(len(input_ids)):
            input_ids[i].insert(0, self.conf.tokenizer.bos_token_id)
        # add eos
        for i in range(len(input_ids)):
            input_ids[i].append(self.conf.tokenizer.eos_token_id)

        ###########
        # IMAGE
        # we use the png files
        img_path = os.path.join(self.conf.img_dir, self.img_split, filename)
        img = cv2_imread(img_path)

        if self.transform:
            _res = self.transform(image=img)
            img = _res['image']

        objs = []
        for i in sort_i:
            each = {}
            for attr in ATTRS:
                each[attr] = group[attr].iloc[i]
            objs.append(each)

        out = {
            'input_ids': input_ids,
            'img': img,
            'filename': filename,
            'text': lines,
            # uncorrupted text with original order
            'text_gt': lines_gt,
            'objs': objs,
        }
        if self.return_index:
            out['index'] = idx + self.index_offset
        return out


class CLEVRTextDataset(CLEVRDataset):
    """same as the above, but each will contain a single long text"""
    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        text = []
        if len(res['input_ids']) == 1:
            text = res['input_ids'][0]
        else:
            for i in range(len(res['input_ids'])):
                if i == 0:
                    # first sentence has <bos> no <eos> with trailing <sep>
                    text += res['input_ids'][i][:-1] + [
                        self.conf.tokenizer.sep_token_id
                    ]
                elif i == len(res['input_ids']) - 1:
                    # last input_ids has no <bos>
                    text += res['input_ids'][i][1:]
                else:
                    # input_ids in between has no <bos> and no <eos> with trailing <sep>
                    text += res['input_ids'][i][1:-1] + [
                        self.conf.tokenizer.sep_token_id
                    ]
        res['input_ids'] = text
        return res


def make_transform(augment, conf: CLEVRTransformConfig):
    inter_opts = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
    }
    inter = inter_opts[conf.interpolation]

    trans = []
    if augment == 'train':
        if conf.rotate > 0:
            trans += [
                Rotate(conf.rotate,
                       border_mode=0,
                       p=conf.rotate_p,
                       interpolation=inter)
            ]
        if conf.crop_p > 0:
            trans += [
                RandomResizedCrop(conf.size,
                                  conf.size,
                                  scale=(conf.min_size, 1.0),
                                  p=conf.crop_p,
                                  interpolation=inter)
            ]
        else:
            trans += [Resize(conf.size, conf.size, interpolation=inter)]

        if conf.bc_prob > 0:
            trans += [
                RandomBrightnessContrast(conf.brightness,
                                         conf.contrast,
                                         p=conf.bc_prob)
            ]

        trans += [Normalize(MEAN, SD)]
    elif augment == 'eval':
        trans += [
            Resize(conf.size, conf.size, interpolation=inter),
            Normalize(MEAN, SD),
        ]
    else:
        raise NotImplementedError()

    trans += [ToTensorV2()]
    return Compose(trans)


class SentencesCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        """
        Return:
        {
            'img': [n, c, h, w],
            'input_ids': [n*m, t],
            'lengths': [n],
        }
        """
        batch = defaultdict(list)

        for each in data:
            for k, v in each.items():
                if k not in ['img', 'input_ids']:
                    batch[k].append(v)

        sent_lengths = [
            len(sentence) for each in data for sentence in each['input_ids']
        ]
        if len(sent_lengths) == 0:
            max_len = None
        else:
            max_len = max(sent_lengths)
        for i in range(len(data)):
            batch['img'].append(data[i]['img'])
            batch['lengths'].append(len(data[i]['text']))
            for sent in data[i]['input_ids']:
                n_pad = max_len - len(sent)
                sent = sent + [self.tokenizer.pad_token_id] * n_pad
                batch['input_ids'].append(sent)

        batch['img'] = torch.stack(batch['img'])
        if len(batch['input_ids']) == 0:
            # keep it 2d
            batch['input_ids'] = torch.empty(0, 0).long()
        else:
            batch['input_ids'] = torch.LongTensor(batch['input_ids'])
        batch['lengths'] = torch.LongTensor(batch['lengths'])
        if 'index' in batch:
            batch['index'] = torch.LongTensor(batch['index'])
        return batch


class TextCollator(SentencesCollator):
    """collator for the long-text dataset"""
    def __call__(self, data):
        """
        Return:
        {
            'img': [n, c, h, w],
            'input_ids': [n, T],
        }
        """
        batch = defaultdict(list)

        for each in data:
            for k, v in each.items():
                if k not in ['img', 'input_ids']:
                    batch[k].append(v)

        max_len = max(len(each['input_ids']) for each in data)
        for i in range(len(data)):
            batch['img'].append(data[i]['img'])
            batch['lengths'].append(len(data[i]['text']))
            n_pad = max_len - len(data[i]['input_ids'])
            input_ids = data[i]['input_ids'] + [self.tokenizer.pad_token_id
                                                ] * n_pad
            batch['input_ids'].append(input_ids)

        batch['img'] = torch.stack(batch['img'])
        batch['input_ids'] = torch.LongTensor(batch['input_ids'])
        batch['lengths'] = torch.LongTensor(batch['lengths'])
        if 'index' in batch:
            batch['index'] = torch.LongTensor(batch['index'])
        return batch


def cv2_imread(path):
    img = cv2.imread(path)
    assert img is not None, f'cannot read {path}'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
