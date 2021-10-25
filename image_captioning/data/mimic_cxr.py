import cv2
from albumentations import *
from albumentations.pytorch import ToTensorV2
from tokenizer import *
from trainer.start import *
from transformers import (BertTokenizerFast, PreTrainedTokenizerFast,
                          RobertaTokenizerFast)

from data.common import BaseDataConfig, BaseDataset
import sacrebleu

cv2.setNumThreads(1)

# chestxray's
MEAN = [0.4984]
SD = [0.2483]

errors = [
    '485f5c59-b289be7c-346a95ca-33143ea7-0c165a1f',
    '5ad2d7d0-cc1732c9-35a8915c-7e77d191-df7b0992',
    '4ec69a3d-c26ee4b8-99157c7d-cad8faaf-4b0fc113',
    'caea02bd-dcd9808d-d596f848-3ddf200a-037dd6d0',
    'd3a2daaa-fb1d9390-3ef22ad2-4a2b246e-65ee5828',
    '0e599b27-310f7dec-73ae46e2-6e450711-22560b32',
    '02798464-0d35345f-c4c6f074-9f2b8fcc-a27d7574',
    '2ddc0f1e-f9f2b601-bebeb8c1-b96d3d8c-60051bdd',
    '45be2b0d-ad5e7b0f-62861dc2-8f7a8f04-32b6d560',
    'b913b602-e420f788-f8ae2e56-b4437002-c1953512',
    '18e010fb-9e40c216-00228d24-5fb7c7c4-6f18edb6',
    '68290e41-12af53f8-160c430e-dca5c9d7-7a9553b9',
    '25806a33-744e8586-0cc8cdc8-ed83d82d-d7dee178',
    'd82fae25-9c068173-624b6e7f-5c5752c5-25f662e5'
]

here = os.path.join(os.path.dirname(__file__), 'mimic_cxr')
report_options = {
    'with_finding': f'{here}/mimic_reports_with_findings_ready.csv',
}

view_options = {
    'pa': f'{here}/mimic-cxr-record-pa.csv',
    'ap': f'{here}/mimic-cxr-record-ap.csv',
    'front': f'{here}/mimic-cxr-record-front.csv',
}
splits = {
    'v1': (
        f'{here}/train.csv',
        f'{here}/val.csv',
        f'{here}/test.csv',
    ),
}


def make_tokenizer(lowercase):
    from pathlib import Path

    from tokenizers import ByteLevelBPETokenizer

    paths = [str(x) for x in Path(f'{here}/reports').glob("**/*.txt")]

    tokenizer = ByteLevelBPETokenizer(lowercase=lowercase)
    tokenizer.train(files=paths,
                    vocab_size=10_000,
                    min_frequency=2,
                    special_tokens=[
                        "<s>",
                        "<pad>",
                        "</s>",
                        "<unk>",
                        "<mask>",
                        '<sep>',
                        "<cls>",
                    ])
    name = 'mimic_text'
    if lowercase:
        name += f'_lowercase'
    tokenizer.save_model('tokenizers', name)


@dataclass
class MimicTransformConfig:
    size: int = 256
    rotate: int = 90
    rotate_prob: float = 0.5
    brightness: float = 0.5
    contrast: float = 0.5
    min_size: float = 0.7
    interpolation: str = 'cubic'

    @property
    def name(self):
        name = f'{self.size}size'
        if self.rotate != 90:
            name += f'-{self.rotate}rot'
        if self.rotate_prob != 0.5:
            name += f'-{self.rotate_prob}rotp'
        if self.brightness != 0.5:
            name += f'-{self.brightness}bright'
        if self.contrast != 0.5:
            name += f'-{self.contrast}cont'
        if self.interpolation != 'cubic':
            name += f'-{self.interpolation}'
        return name

    def make_transform(self, augment):
        inter_opts = {
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
        }
        inter = inter_opts[self.interpolation]

        trans = []
        if augment == 'train':
            if self.rotate > 0:
                trans += [
                    Rotate(self.rotate,
                           border_mode=0,
                           p=self.rotate_prob,
                           interpolation=inter)
                ]
            if self.min_size == 1:
                trans += [Resize(self.size, self.size, interpolation=inter)]
            else:
                trans += [
                    RandomResizedCrop(self.size,
                                      self.size,
                                      scale=(self.min_size, 1.0),
                                      p=1.0,
                                      interpolation=inter)
                ]
            trans += [HorizontalFlip(p=0.5)]
            if self.contrast > 0 or self.brightness > 0:
                trans += [
                    RandomBrightnessContrast(self.brightness,
                                             self.contrast,
                                             p=0.5)
                ]
            trans += [Normalize(MEAN, SD)]
        elif augment == 'eval':
            trans += [
                Resize(self.size, self.size, interpolation=inter),
                Normalize(MEAN, SD),
            ]
        else:
            raise NotImplementedError()

        trans += [GrayToTensor()]
        return Compose(trans)


@dataclass
class MimicTextDatasetConfig(BaseDataConfig):
    tokenizer_type: str = 'roberta'
    tokenizer: PreTrainedTokenizerFast = None
    n_max_sentences: int = 10
    n_max_sentence_length: int = 40
    do_sample: bool = True
    split: str = 'v1'
    order_by: str = 'none'
    report: str = 'with_finding'
    view: str = 'front'
    cxr_base_dir = f'{here}/images512'
    do_pad: bool = False
    lower_case: bool = True
    trans_conf: MimicTransformConfig = MimicTransformConfig()

    @property
    def name(self):
        name = f'bs{self.batch_size}_mimictext-{self.split}-{self.report}-{self.view}-{self.order_by}-max({self.n_max_sentences},{self.n_max_sentence_length})'
        if self.do_sample and self.n_max_sentences is not None:
            name += f'-samp'
        if self.do_pad:
            name += f'-pad'
        if self.lower_case:
            name += f'-lower'
        name += f'-{self.tokenizer_type}'
        name += f'_{self.trans_conf.name}'
        return name

    def make_dataset(self):
        self.make_tokenizer()
        return MimicTextCombinedDataset(self)

    def make_tokenizer(self):
        if self.tokenizer_type == 'roberta':
            if self.lower_case:
                self.tokenizer = load_roberta_tokenizer('mimic_text_lowercase')
            else:
                self.tokenizer = load_roberta_tokenizer('mimic_text')
        else:
            raise NotImplementedError()
        return self.tokenizer


class MimicTextCombinedDataset(BaseDataset):
    def __init__(self, conf: MimicTextDatasetConfig):
        self.conf = conf

        train, val, test = splits[conf.split]
        if conf.trans_conf is not None:
            trans_train = conf.trans_conf.make_transform('train')
            trans_eval = conf.trans_conf.make_transform('eval')
        else:
            trans_train = None
            trans_eval = None
        self.train = MimicTextDataset(train, conf, trans_train)
        self.val = MimicTextDataset(val, conf, trans_eval)
        # eval mode keep all sentences
        conf2 = conf.clone()
        conf2.n_max_sentence_length = None
        conf2.n_max_sentences = None
        conf2.do_pad = False
        self.val_eval = MimicTextDataset(val, conf2, trans_eval)
        self.test = MimicTextDataset(test, conf2, trans_eval)

    def make_loader(self, data, shuffle):
        return DataLoader(
            data,
            batch_size=self.conf.batch_size,
            collate_fn=MimicTextCollator(self.conf.tokenizer),
            shuffle=shuffle,
            pin_memory=self.conf.pin_memory,
            num_workers=self.conf.num_workers,
            multiprocessing_context=(mp.get_context('fork')
                                     if self.conf.num_workers > 0 else None),
        )

    @property
    def metric(self):
        return [
            'bleu4_micro_f1', 'bleu4_micro_recall', 'bleu4_micro_precision'
        ]

    @classmethod
    def evaluate_instance(cls, pred: List[str], gt: List[str]):
        scores = best_report_bleu_chamfer(gt, pred)
        return scores

    @classmethod
    def evaluate(cls,
                 pred: List[List[str]],
                 gt: List[List[str]],
                 progress=False):
        assert len(pred) == len(gt)
        return best_sentence_level_bleu(gt, pred, progress=progress)


class MimicTextDataset(Dataset):
    """
    Args:
        study_id_csv: columns "study_id"
        report_csv: columns "study_id", "text"
        record_csv: columns "study_id", "path"
    """
    def __init__(
        self,
        study_id_csv,
        conf: MimicTextDatasetConfig,
        transform=None,
    ):
        self.conf = conf
        # make the df
        study_id_df = pd.read_csv(study_id_csv)
        report_csv = report_options[conf.report]
        report_df = pd.read_csv(report_csv)
        view_csv = view_options[conf.view]
        df = pd.read_csv(view_csv)

        # select only mentioned
        df = df[df['study_id'].isin(study_id_df['study_id'])]
        # select only those we have the reports
        report_study_id = set(report_df['study_id'])
        df = df[df['study_id'].isin(report_study_id)]
        # select only we have readable images
        df = df[~df['dicom_id'].isin(errors)].reset_index(drop=True)

        self.report_df = report_df
        self.record_df = df

        self.transform = transform

    def __len__(self):
        return len(self.record_df)

    def __getitem__(self, idx):
        ############
        # REPORT
        study_id = self.record_df.loc[idx, 'study_id']
        text = self.report_df[self.report_df['study_id'] ==
                              study_id].iloc[0]['text']
        if text != text:
            # nan = empty report
            text = ''

        if self.conf.lower_case:
            text = text.lower()

        lines = text.split('\n')

        if self.conf.do_sample and self.conf.n_max_sentences is not None:
            if self.conf.n_max_sentences < len(lines):
                # shuffle before selecting only subset
                random.shuffle(lines)
                lines = lines[:self.conf.n_max_sentences]

        if self.conf.order_by == 'none':
            pass
        elif self.conf.order_by == 'shortest':
            lines = sorted(lines, key=lambda x: len(x))
        elif self.conf.order_by == 'alphabet':
            lines = sorted(lines)
        elif self.conf.order_by == 'random':
            random.shuffle(lines)
        else:
            raise NotImplementedError()

        if self.conf.do_pad:
            n_pad = self.conf.n_max_sentences - len(lines)
            lines += [''] * n_pad

        if not self.conf.do_sample and self.conf.n_max_sentences is not None:
            lines = lines[:self.conf.n_max_sentences]

        assert self.conf.tokenizer is not None
        if isinstance(self.conf.tokenizer, BertTokenizerFast):
            # bert tokenizer bug
            # [''] => error
            # so we need to force add special tokens
            res = self.conf.tokenizer(lines,
                                      return_attention_mask=False,
                                      add_special_tokens=True)
        elif isinstance(self.conf.tokenizer, RobertaTokenizerFast):
            res = self.conf.tokenizer(lines,
                                      return_attention_mask=False,
                                      add_special_tokens=False)
        else:
            raise NotImplementedError()

        input_ids = res['input_ids']

        if isinstance(self.conf.tokenizer, BertTokenizerFast):
            # trim extra lengths
            if self.conf.n_max_sentence_length is not None:
                for i in range(len(input_ids)):
                    # leave room for <eos> token
                    input_ids[i] = input_ids[i][:self.conf.
                                                n_max_sentence_length - 1]
        elif isinstance(self.conf.tokenizer, RobertaTokenizerFast):
            # add the <bos> token
            input_ids = add_bos_token(input_ids,
                                      self.conf.tokenizer.bos_token_id)
            # trim extra lengths
            if self.conf.n_max_sentence_length is not None:
                for i in range(len(input_ids)):
                    # leave room for <eos> token
                    input_ids[i] = input_ids[i][:self.conf.
                                                n_max_sentence_length - 1]
            # add the <eos> token
            input_ids = add_eos_token(input_ids,
                                      self.conf.tokenizer.eos_token_id)
        else:
            raise NotImplementedError()

        ###########
        # IMAGE
        # we use the png files
        img_path = self.record_df.loc[idx, 'path'].replace('.dcm', '.png')
        # remove the prefix files/
        img_path = img_path.replace('files/', '')
        img_path = f'{self.conf.cxr_base_dir}/{img_path}'
        img = cv2_loader(img_path)

        if self.transform:
            _res = self.transform(image=img)
            img = _res['image']

        return {
            'input_ids': input_ids,
            'is_pad':
            torch.BoolTensor([lines[i] == '' for i in range(len(lines))]),
            'img': img,
            'study_id': study_id,
            'text': lines,
            'index': idx,
        }


@dataclass
class MimicTextTextDatasetConfig(MimicTextDatasetConfig):
    def make_dataset(self):
        return MimicTextTextCombinedDataset(self)


class MimicTextTextCombinedDataset(MimicTextCombinedDataset):
    def __init__(self, conf: MimicTextDatasetConfig):
        self.conf = conf

        train, val, test = splits[conf.split]
        self.train = MimicTextTextDataset(
            train, conf, conf.trans_conf.make_transform('train'))
        self.val = MimicTextTextDataset(val, conf,
                                        conf.trans_conf.make_transform('eval'))
        # eval mode keep all sentences
        conf2 = conf.clone()
        conf2.n_max_sentence_length = None
        conf2.n_max_sentences = None
        conf2.do_pad = False
        self.val_eval = MimicTextTextDataset(
            val, conf2, conf.trans_conf.make_transform('eval'))
        self.test = MimicTextTextDataset(
            test, conf2, conf.trans_conf.make_transform('eval'))

    def make_loader(self, data, shuffle):
        return DataLoader(
            data,
            batch_size=self.conf.batch_size,
            collate_fn=MimicTextTextCollator(self.conf.tokenizer),
            shuffle=shuffle,
            pin_memory=self.conf.pin_memory,
            num_workers=self.conf.num_workers,
            multiprocessing_context=(mp.get_context('fork')
                                     if self.conf.num_workers > 0 else None),
        )


class MimicTextTextDataset(MimicTextDataset):
    """same as the above, but each will contain a single long text"""
    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        text = []
        if len(res['input_ids']) == 0:
            # empty report
            text = [
                self.conf.tokenizer.bos_token_id,
                self.conf.tokenizer.eos_token_id
            ]
        elif len(res['input_ids']) == 1:
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
        return {
            **res,
            'input_ids': text,
        }


class MimicTextCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        out = defaultdict(list)

        sent_lengths = [
            len(input_ids) for each in data for input_ids in each['input_ids']
        ]
        if len(sent_lengths) == 0:
            max_len = None
        else:
            max_len = max(sent_lengths)
        for i in range(len(data)):
            out['img'].append(data[i]['img'])
            out['lengths'].append(len(data[i]['input_ids']))
            out['text'].append(data[i]['text'])
            out['study_id'].append(data[i]['study_id'])
            out['index'].append(data[i]['index'])
            for sent in data[i]['input_ids']:
                n_pad = max_len - len(sent)
                sent = sent + [self.tokenizer.pad_token_id] * n_pad
                out['input_ids'].append(sent)
            out['is_pad'].append(data[i]['is_pad'])

        out['img'] = torch.stack(out['img'])
        if len(out['input_ids']) == 0:
            # keep it 2d
            out['input_ids'] = torch.empty(0, 0).long()
        else:
            out['input_ids'] = torch.LongTensor(out['input_ids'])
        out['lengths'] = torch.LongTensor(out['lengths'])
        out['index'] = torch.LongTensor(out['index'])
        out['is_pad'] = torch.cat(out['is_pad'])
        return out


class MimicTextTextCollator:
    """collator for the long-text dataset"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        out = defaultdict(list)

        sent_lengths = [len(each['input_ids']) for each in data]
        if len(sent_lengths) == 0:
            max_len = None
        else:
            max_len = max(sent_lengths)
        for i in range(len(data)):
            out['img'].append(data[i]['img'])
            out['lengths'].append(len(data[i]['text']))
            out['text'].append(data[i]['text'])
            out['study_id'].append(data[i]['study_id'])
            out['index'].append(data[i]['index'])

            ids = data[i]['input_ids']
            n_pad = max_len - len(ids)
            ids = ids + [self.tokenizer.pad_token_id] * n_pad
            out['input_ids'].append(ids)

        out['img'] = torch.stack(out['img'])
        if len(out['input_ids']) == 0:
            # keep it 2d
            out['input_ids'] = torch.empty(0, 0).long()
        else:
            out['input_ids'] = torch.LongTensor(out['input_ids'])
        out['lengths'] = torch.LongTensor(out['lengths'])
        out['index'] = torch.LongTensor(out['index'])
        return out


class GrayToTensor(ToTensorV2):
    def apply(self, img, **params):
        # because of gray scale
        # we add an additional channel
        return torch.from_numpy(img).unsqueeze(0)


def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f'cannot load {path}'
    return img


def add_bos_token(sentences, bos_token_id):
    out = []
    for each in sentences:
        out.append([bos_token_id] + each)
    return out


def add_eos_token(sentences, eos_token_id):
    out = []
    for each in sentences:
        if each[-1] != eos_token_id:
            each = each + [eos_token_id]
        out.append(each)
    return out


class MimicTextEvaluator(CollectCb):
    def __init__(self, keys):
        # collect 'pred' and 'y'
        super().__init__(keys=keys)
        self.keys = keys

    def on_ep_end(self, buffer, i_itr, **kwargs):
        pred = buffer[self.keys[0]]
        y = buffer[self.keys[1]]
        metric = MimicTextCombinedDataset.evaluate(pred, y)
        info = {
            'i_itr': i_itr,
            **metric,
        }
        self.add_to_bar_and_hist(info)
        self._flush()


class MacroAverage:
    def __init__(self) -> None:
        self.items = []

    def add(self, items):
        self.items.append(np.array(items).mean())

    def val(self):
        return np.array(self.items).mean()


class WeightedAverage:
    def __init__(self):
        self.items = []

    def add(self, items):
        self.items += items

    def val(self):
        return np.array(self.items).mean()


def f1_score(prec, rec):
    return 2 * (prec * rec) / ((prec + rec) + 1e-8)


def best_sentence_level_bleu(ref_reports, hyp_reports, progress=False):
    """
    Args:
        references: list of list of list of sentences [report[sentence]]
        hypotheses: list of list of sentences [report[sentence]]
    """
    micro = defaultdict(WeightedAverage)

    itr = range(len(ref_reports))
    if progress:
        itr = tqdm(itr)
    for i in itr:
        ref_report = ref_reports[i]
        hyp_report = hyp_reports[i]
        scores = bleu_all_pair(ref_report, hyp_report)
        # chamfer
        recall, precision = best_report_bleu_chamfer(ref_report, hyp_report,
                                                     scores)
        micro['recall'].add(list(recall))
        micro['precision'].add(list(precision))

    micro_recall = micro['recall'].val()
    micro_prec = micro['precision'].val()
    micro_f1 = f1_score(micro_prec, micro_recall)

    return {
        'bleu4_micro_f1': micro_f1,
        'bleu4_micro_recall': micro_recall,
        'bleu4_micro_precision': micro_prec,
    }


def bleu_all_pair(references, predicts):
    scores = []
    for ref in references:
        row = []
        for pred in predicts:
            # ref: Post, Matt. 2018. “A Call for Clarity in Reporting BLEU Scores.” In Proceedings of the Third Conference on Machine Translation: Research Papers, 186–91. Brussels, Belgium: Association for Computational Linguistics.
            # read: https://github.com/mjpost/sacrebleu/blob/5dfcaa3cee00039bcad7a12147b6d6976cb46f42/sacrebleu/metrics/bleu.py#L46
            if ref == '':
                score = 100. if pred == '' else 0.
            else:
                score = sacrebleu.sentence_bleu(pred, [ref]).score
            row.append(score)
        scores.append(row)
    # (N, M)
    scores = np.array(scores)
    return scores


def best_report_bleu_chamfer(references, predicts, scores=None):
    """best chamfer match bleu score"""
    if scores is None:
        scores = bleu_all_pair(references, predicts)

    # recall
    ref_to_pred = scores.max(axis=1)
    # precision
    pred_to_ref = scores.max(axis=0)
    return ref_to_pred, pred_to_ref
