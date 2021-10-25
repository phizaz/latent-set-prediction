from model.cnn_text import *
from trainer.start import *
import pickle

from data.mimic_cxr import *
from model.cnn_text_set import *
from model.cnn_text_seq import *
from utils.optimizer import *
from train_clevr import ImgTextTrainer
import cv2

torch.set_num_threads(1)
cv2.setNumThreads(1)

UnionDataConfig = Union[MimicTextDatasetConfig, MimicTextTextDatasetConfig]
UnionOptConfig = Union[OptimizerConfig, AsymOptimizerConfig]


@dataclass
class CXRTextConfig(BaseConfig):
    device: str = None
    seed: int = 0
    n_ep: int = 60
    data_conf: UnionDataConfig = None
    net_conf: CNNTextSetModelConfig = None
    opt_conf: UnionOptConfig = OptimizerConfig()
    fp16: bool = True
    debug: bool = False
    do_save: bool = True
    resume: bool = True
    eval_small_percent: float = 0.3
    write_to_file: bool = False
    load_dir: str = 'best'
    resume_from: str = 'last'
    track_switches: bool = False

    @property
    def name(self):
        name = []
        name.append(self.data_conf.name)
        name.append(self.net_conf.name)
        tmp = self.opt_conf.name
        if self.fp16:
            tmp += f'_fp16'
        if self.track_switches:
            tmp += f'-track'
        name.append(tmp)
        name.append(f'{self.seed}')
        return '/'.join(name)


class CXRTextTrainer(ImgTextTrainer):
    def __init__(self, conf: CXRTextConfig, data: BaseDataset):
        super().__init__(conf, data)


class ImgTextExperiment:
    def __init__(self, conf: CXRTextConfig) -> None:
        self.conf = conf
        self.make_data()
        self.trainer_cls = CXRTextTrainer
        if conf.fp16:
            self.trainer_cls = amp_trainer_mask(self.trainer_cls)

    def make_data(self):
        self.data = self.conf.data_conf.make_dataset()
        self.train_loader = ConvertLoader(
            self.data.make_loader(self.data.train, shuffle=True),
            device=self.conf.device,
        )
        self.val_loader = ConvertLoader(
            self.data.make_loader(self.data.val, shuffle=False),
            device=self.conf.device,
        )
        self.val_small_loader = ConvertLoader(
            self.data.make_loader(
                SubsetDataset(self.data.val_eval,
                              size=int(
                                  len(self.data.val_eval) *
                                  self.conf.eval_small_percent)),
                shuffle=False,
            ),
            device=self.conf.device,
        )
        self.test_loader = ConvertLoader(
            self.data.make_loader(self.data.test, shuffle=False),
            device=self.conf.device,
        )

    def load_trainer(self):
        trainer = self.trainer_cls(self.conf, self.data)
        trainer.load(f'save/{self.conf.name}/checkpoints/{self.conf.load_dir}')
        return trainer

    def train(self):
        set_seed(self.conf.seed)
        trainer = self.trainer_cls(self.conf, self.data)
        callbacks = trainer.make_default_callbacks()
        callbacks += [
            # eval with teacher forcing
            ValidateCb(self.val_loader,
                       callbacks=[AvgCb(trainer.metrics)],
                       name='val',
                       n_ep_cycle=self.conf.opt_conf.n_ep_eval_cycle),
            # eval without teacher forcing
            ValidateCb(self.val_small_loader,
                       callbacks=[
                           AvgCb(trainer.metrics),
                           MimicTextEvaluator(['pred_str', 'text'])
                       ],
                       name='val_real',
                       n_ep_cycle=self.conf.opt_conf.n_ep_eval_cycle,
                       predictor_cls=RealValidatePredictor),
        ]
        callbacks += self.conf.opt_conf.make_scheduler()
        if self.conf.do_save:
            callbacks += [
                LiveDataframeCb(f'save/{self.conf.name}/stats.csv'),
                AutoResumeCb(
                    f'save/{self.conf.name}/checkpoints',
                    n_ep_cycle=self.conf.opt_conf.n_ep_eval_cycle,
                    keep_best=True,
                    metric=self.conf.opt_conf.best_metric,
                    metric_best=self.conf.opt_conf.best_mode,
                    resume=self.conf.resume,
                    resume_from=self.conf.resume_from,
                ),
            ]
        trainer.train(self.train_loader,
                      n_max_ep=self.conf.n_ep,
                      callbacks=callbacks)

    def test(self):
        trainer = self.load_trainer()
        predictor = RealValidatePredictor(
            trainer,
            callbacks=[
                ProgressCb('test'),
                MimicTextEvaluator(['pred_str', 'text'])
            ],
        )
        out, extras = predictor.predict(self.test_loader)
        out.update(extras)
        print(out)

        path = f'eval/{self.conf.name}.csv'
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        df = DataFrame([out])
        df.to_csv(path, index=False)

        with open(f'eval/{self.conf.name}.pkl', 'wb') as f:
            pickle.dump(predictor.buffer, f)

    def warm_dataset(self):
        for each in tqdm(self.train_loader):
            pass
        for each in tqdm(self.val_loader):
            pass
        for each in tqdm(self.test_loader):
            pass


def opt_conf(lr, lr2, scheduler, best_metric, rop_patience):
    if lr2 is None:
        opt_conf = OptimizerConfig(n_ep_eval_cycle=1,
                                   scheduler=scheduler,
                                   best_metric=best_metric,
                                   rop_patience=rop_patience,
                                   lr=lr)
    else:
        opt_conf = AsymOptimizerConfig(n_ep_eval_cycle=1,
                                       scheduler=scheduler,
                                       best_metric=best_metric,
                                       rop_patience=rop_patience,
                                       lr=lr,
                                       lr2=lr2)
    return opt_conf


def clevr_set(
    type,
    seed,
    order_by='none',
    do_sample=True,
    n_max_length=40,
    n_max_sentences=10,
    do_pad=True,
    batch_size=64,
    lr=1e-4,
    lr2=None,
    n_hid=256,
    n_head=4,
    n_set_layer=3,
    n_layer=3,
    n_trans_layer=3,
    w_loss_br=1,
    w_loss_rb=0.1,
    w_empty=None,
    gcr_mode='gcr',
    safe_coef=0,
    trans_type='transformer',
    use_cache=False,
    dict_type='sin',
    pos_emb_type='sin',
    backbone='resnet34',
    best_metric='val_loss_tgt',
    loss_balance_eos=False,
    rop_patience=1,
    report='with_finding',
    scheduler='rop',
    track_switches=False,
    dropout=0,
):
    data_conf = MimicTextDatasetConfig(
        batch_size=batch_size,
        order_by=order_by,
        do_sample=do_sample,
        num_workers=ENV.num_workers,
        trans_conf=MimicTransformConfig(),
        do_pad=do_pad,
        n_max_sentences=n_max_sentences,
        n_max_sentence_length=n_max_length,
        report=report,
    )
    tokenizer = data_conf.make_tokenizer()

    dec_conf = TextPytorchDecoderConfig(
        tokenizer=tokenizer,
        n_max_length=n_max_length,
        n_hid=n_hid,
        n_head=n_head,
        n_ff=n_hid * 4,
        n_layer=n_layer,
        use_cache=use_cache,
        pos_emb_type=pos_emb_type,
        loss_balance_eos=loss_balance_eos,
        w_empty=w_empty,
        dropout=dropout,
    )

    if type == 'set':
        loss_conf = MSEGCRLatentLossConfig(
            w_loss_br=w_loss_br,
            w_loss_rb=w_loss_rb,
            gcr_mode=gcr_mode,
            safe_coef=safe_coef,
        )

        text_enc_conf = TextEncoderWithContextConfig(
            tokenizer,
            n_max_length=n_max_length,
            n_hid=n_hid,
            n_head=n_head,
            n_ff=n_hid * 4,
            n_layer=n_layer,
            pos_emb_type=pos_emb_type,
            dropout=dropout,
        )

        set_dec_conf = SetDecoderVarySizeConfig(
            n_max_items=n_max_sentences,
            n_hid=n_hid,
            dict_type=dict_type,
            n_head=n_head,
            n_ff=n_hid * 4,
            n_layer=n_set_layer,
            loss_conf=loss_conf,
            dropout=dropout,
        )

        net_conf = CNNTextSetModelConfig(
            set_dec_conf=set_dec_conf,
            text_enc_conf=text_enc_conf,
            text_dec_conf=dec_conf,
            backbone=backbone,
            n_in=1,
            n_att=n_hid,
            trans_type=trans_type,
            tf_n_ff=n_hid * 4,
            tf_n_head=n_head,
            tf_n_layer=n_trans_layer,
            tf_dropout=dropout,
        )
    elif type == 'seq':
        net_conf = CNNTextSeqModelConfig(
            seq_dec_conf=SeqDecoderVarySizeConfig(
                n_max_items=n_max_sentences,
                n_hid=n_hid,
                dict_type=dict_type,
                n_head=n_head,
                n_ff=n_hid * 4,
                n_layer=n_set_layer,
                dropout=dropout,
            ),
            text_dec_conf=dec_conf,
            backbone=backbone,
            n_in=1,
            n_att=n_hid,
            trans_type=trans_type,
            tf_n_ff=n_hid * 4,
            tf_n_head=n_head,
            tf_n_layer=n_trans_layer,
            tf_dropout=dropout,
        )
    else:
        raise NotImplementedError()

    conf = CXRTextConfig(
        seed=seed,
        data_conf=data_conf,
        net_conf=net_conf,
        opt_conf=opt_conf(lr, lr2, scheduler, best_metric, rop_patience),
        track_switches=track_switches,
    )
    return conf


def clevr_text(
    seed=0,
    batch_size=64,
    order_by='none',
    do_sample=True,
    n_max_length=40,
    n_max_sentences=10,
    lr=1e-4,
    lr2=None,
    n_hid=256,
    n_head=4,
    n_layer=3,
    n_trans_layer=3,
    text_dec_type='pytorch',
    trans_type='transformer',
    use_cache=False,
    pos_emb_type='sin',
    backbone='resnet34',
    best_metric='val_loss',
    loss_balance_eos=False,
    rop_patience=1,
    report='with_finding',
    scheduler='rop',
    dropout=0,
):
    data_conf = MimicTextTextDatasetConfig(
        batch_size=batch_size,
        order_by=order_by,
        num_workers=ENV.num_workers,
        trans_conf=MimicTransformConfig(),
        do_sample=do_sample,
        do_pad=False,
        n_max_sentences=n_max_sentences,
        n_max_sentence_length=n_max_length,
        report=report,
    )
    tokenizer = data_conf.make_tokenizer()

    dec_conf = TextPytorchDecoderConfig(
        tokenizer=tokenizer,
        n_max_length=n_max_length * n_max_sentences,
        n_hid=n_hid,
        n_head=n_head,
        n_ff=n_hid * 4,
        n_layer=n_layer,
        use_cache=use_cache,
        pos_emb_type=pos_emb_type,
        loss_balance_eos=loss_balance_eos,
        dropout=dropout,
    )

    conf = CXRTextConfig(
        seed=seed,
        data_conf=data_conf,
        net_conf=CNNTextModelConfig(
            text_dec_conf=dec_conf,
            backbone=backbone,
            n_in=1,
            n_att=n_hid,
            trans_type=trans_type,
            tf_n_ff=n_hid * 4,
            tf_n_head=n_head,
            tf_n_layer=n_trans_layer,
            tf_dropout=dropout,
        ),
        opt_conf=opt_conf(lr, lr2, scheduler, best_metric, rop_patience),
    )
    return conf


class Run:
    def __init__(self, namespace: str = ''):
        self.namespace = namespace

    def __call__(self, conf: CXRTextConfig):
        # conf.debug = True
        # conf.do_save = False
        # conf.resume = False
        conf.write_to_file = True
        # conf.resume_from = 'best'
        with global_queue(enable=not conf.debug, namespace=self.namespace):
            with cuda_round_robin(enable=not conf.debug,
                                  namespace=self.namespace) as conf.device:
                with redirect_to_file(enable=conf.write_to_file):
                    print(conf.name)
                    exp = ImgTextExperiment(conf)
                    # exp.warm_dataset()
                    exp.train()
                    exp.test()
