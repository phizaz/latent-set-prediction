from model.cnn_text import *
from trainer.start import *

from data.clevr import *
from model.cnn_text_set import *
from model.cnn_text_seq import *
from utils.optimizer import *
import cv2

torch.set_num_threads(1)
cv2.setNumThreads(1)


@dataclass
class ImgTextConfig(BaseConfig):
    device: str = None
    seed: int = 0
    n_ep: int = 40
    data_conf: CLEVRDataConfig = None
    net_conf: CNNTextSetModelConfig = None
    opt_conf: OptimizerConfig = OptimizerConfig()
    fp16: bool = True
    debug: bool = False
    do_save: bool = True
    resume: bool = True
    eval_small_percent: float = 0.3
    write_to_file: bool = False
    load_dir: str = 'best'
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


class SwitchCountCb(StatsCallback):
    def __init__(self, n_row, n_col):
        super().__init__(n_log_cycle=1)
        self.n_col = n_col
        self._state['data'] = torch.full((n_row, n_col), -1).long()

    def load(self, path: str, map_location=None):
        """load state from file"""
        if self.is_state_empty():
            # this cb doesn't need a state
            # caution: cb that needs a state must have the "footprint"
            # of the states, so that it would not be empty at first!
            # unless it will not be loaded!
            return
        self.load_state(torch.load(path, map_location='cpu'))

    def on_forward_end(self, forward, i_itr, **kwargs):
        index, lengths, ordering = forward['index'], forward[
            'lengths'], forward['ordering']
        assert len(index) == len(lengths)
        # (n, col)
        new = chunk_by_lengths(ordering.clone(), lengths)
        offset = 0
        for i in range(len(lengths)):
            new[i] -= offset
            offset += lengths[i].item()
        tot_changes = 0
        for i in range(len(lengths)):
            changes = (self.data[index[i], :lengths[i]] != new[i]).sum().item()
            self.data[index[i], :lengths[i]] = new[i]
            tot_changes += changes
        # switches / location / instance
        self.add_to_bar_and_hist({
            'i_itr': i_itr,
            'switches': tot_changes / lengths.sum().item()
        })


class ImgTextTrainer(BaseTrainer):
    def __init__(self, conf: ImgTextConfig, data: BaseDataset):
        super().__init__(conf)
        self.conf = conf
        self.data = data

    @property
    def metrics(self):
        if isinstance(self.conf.net_conf, CNNTextModelConfig):
            return [
                'loss',
                'acc_length',
                'rmse_length',
            ]
        elif isinstance(self.conf.net_conf, CNNTextSeqModelConfig):
            return ['loss_tgt', 'loss', 'rmse_length']
        elif isinstance(self.conf.net_conf, CNNTextSetModelConfig):
            return ['loss_tgt', 'loss_BR', 'loss', 'rmse_length']
        else:
            raise NotImplementedError()

    def forward_pass(self, data, i_itr, **kwargs):
        with time_elapsed_to_profiler('forward', i_itr=i_itr):
            out = self.net.forward(**data, labels=data['input_ids'])
        return {
            **data,
            **out.__dict__,
            'n': len(data['img']),
        }

    @torch.no_grad()
    def eval_forward_pass(self, data, **kwargs) -> Dict:
        with set_mode(self.net, 'eval'):
            pred_str = self.net.eval_forward(img=data['img'])
        dev = data['img'].device
        pred_lengths = torch.LongTensor([len(each)
                                         for each in pred_str]).to(dev)
        lengths = data['lengths']
        acc_length = (pred_lengths == lengths).float().mean()
        rmse_length = (pred_lengths - lengths).float().pow(2).sqrt().mean()
        return {
            **data,
            'pred_str': pred_str,
            'acc_length': acc_length,
            'rmse_length': rmse_length,
            'n': len(data['img']),
        }

    def make_net(self):
        return self.conf.net_conf.make_model()

    def make_opt(self, net):
        return self.conf.opt_conf.make_opt(net)

    def make_default_callbacks(self):
        cb = super().make_default_callbacks() + [MovingAvgCb(self.metrics)]
        if self.conf.track_switches:
            n_row = len(self.data.train)
            n_col = self.conf.data_conf.n_max_sentences
            cb.append(SwitchCountCb(n_row, n_col))
        return cb


class ImgTextExperiment:
    def __init__(self, conf: ImgTextConfig) -> None:
        self.conf = conf
        self.make_data()
        self.trainer_cls = ImgTextTrainer
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
                SubsetDataset(self.data.val,
                              size=int(
                                  len(self.data.val) *
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
                           CLEVRMetricCb(['pred_str', 'objs'])
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
                CLEVRMetricCb(['pred_str', 'objs'])
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


def text_dec_conf(tokenizer, n_max_length, n_hid, n_head, n_layer, use_cache,
                  pos_emb_type, loss_balance_eos):
    text_dec_conf = TextPytorchDecoderConfig(
        tokenizer=tokenizer,
        n_max_length=n_max_length,
        n_hid=n_hid,
        n_head=n_head,
        n_ff=n_hid * 4,
        n_layer=n_layer,
        use_cache=use_cache,
        pos_emb_type=pos_emb_type,
        loss_balance_eos=loss_balance_eos,
    )
    return text_dec_conf


def clevr_set(
    type,
    seed,
    split='clevr',
    do_pad=True,
    batch_size=64,
    lr=1e-4,
    n_hid=256,
    n_head=4,
    n_set_layer=3,
    n_layer=3,
    n_trans_layer=3,
    w_loss_br=1,
    w_loss_rb=0.1,
    gcr_mode='gcr',
    safe_coef=0,
    trans_type='conv1',
    use_cache=False,
    dict_type='sin',
    pos_emb_type='sin',
    backbone='resnet34',
    best_metric='val_loss_tgt',
    loss_balance_eos=False,
    rop_patience=1,
    track_switches=False,
):
    tokenizer = load_roberta_tokenizer('clevr')
    n_max_length = 10
    n_max_sentences = 10

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
    )

    if type == 'set':
        loss_conf = MSEGCRLatentLossConfig(
            w_loss_br=w_loss_br,
            w_loss_rb=w_loss_rb,
            gcr_mode=gcr_mode,
            equal_instance_weight=True,
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
        )

        net_conf = CNNTextSetModelConfig(
            set_dec_conf=SetDecoderVarySizeConfig(
                n_max_items=n_max_sentences,
                n_hid=n_hid,
                dict_type=dict_type,
                n_head=n_head,
                n_ff=n_hid * 4,
                n_layer=n_set_layer,
                loss_conf=loss_conf,
            ),
            text_enc_conf=text_enc_conf,
            text_dec_conf=dec_conf,
            backbone=backbone,
            n_att=n_hid,
            trans_type=trans_type,
            tf_n_ff=n_hid * 4,
            tf_n_head=n_head,
            tf_n_layer=n_trans_layer,
        )
    elif type == 'seq':
        net_conf = CNNTextSeqModelConfig(
            seq_dec_conf=SeqDecoderVarySizeConfig(
                n_max_items=n_max_sentences,
                n_hid=n_hid,
                dict_type=dict_type,
                n_num_element_layer=1,
                n_head=n_head,
                n_ff=n_hid * 4,
                n_layer=n_set_layer,
            ),
            text_dec_conf=dec_conf,
            backbone=backbone,
            n_att=n_hid,
            trans_type=trans_type,
            tf_n_ff=n_hid * 4,
            tf_n_head=n_head,
            tf_n_layer=n_trans_layer,
        )
    else:
        raise NotImplementedError()

    conf = ImgTextConfig(
        seed=seed,
        data_conf=CLEVRDataConfig(
            batch_size=batch_size,
            split=split,
            num_workers=ENV.num_workers,
            tokenizer=tokenizer,
            trans_conf=CLEVRTransformConfig(),
            do_pad=do_pad,
            n_max_sentences=n_max_sentences,
        ),
        net_conf=net_conf,
        opt_conf=OptimizerConfig(n_ep_eval_cycle=1,
                                 best_metric=best_metric,
                                 rop_patience=rop_patience,
                                 lr=lr),
        track_switches=track_switches,
    )
    return conf


def clevr_text(
    split='clevr',
    seed=0,
    batch_size=64,
    lr=1e-4,
    n_hid=256,
    n_head=4,
    n_layer=3,
    n_trans_layer=3,
    trans_type='conv1',
    use_cache=False,
    pos_emb_type='sin',
    backbone='resnet34',
    loss_balance_eos=False,
    rop_patience=1,
):
    tokenizer = load_roberta_tokenizer('clevr')
    n_max_length = 100

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
    )

    conf = ImgTextConfig(
        seed=seed,
        data_conf=CLEVRDataConfig(
            batch_size=batch_size,
            split=split,
            do_pad=False,
            num_workers=ENV.num_workers,
            type='text',
            tokenizer=tokenizer,
            trans_conf=CLEVRTransformConfig(),
        ),
        net_conf=CNNTextModelConfig(
            text_dec_conf=dec_conf,
            backbone=backbone,
            n_att=n_hid,
            trans_type=trans_type,
            tf_n_ff=n_hid * 4,
            tf_n_head=n_head,
            tf_n_layer=n_trans_layer,
        ),
        opt_conf=OptimizerConfig(n_ep_eval_cycle=1,
                                 best_metric='val_loss',
                                 rop_patience=rop_patience,
                                 lr=lr),
    )
    return conf


class Run:
    def __init__(self, namespace: str = ''):
        self.namespace = namespace

    def __call__(self, conf: ImgTextConfig):
        # conf.debug = True
        # conf.do_save = False
        # conf.resume = False
        conf.write_to_file = True
        with global_queue(enable=not conf.debug, namespace=self.namespace):
            with cuda_round_robin(enable=not conf.debug,
                                  namespace=self.namespace) as conf.device:
                with redirect_to_file(enable=conf.write_to_file):
                    print(conf.name)
                    exp = ImgTextExperiment(conf)
                    # exp.warm_dataset()
                    exp.train()
                    exp.test()
