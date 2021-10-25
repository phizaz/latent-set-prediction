from itertools import chain

from segmentation_models_pytorch.encoders import get_encoder
from trainer.start import *

from .cnn_text_set import CNNTextSetModel, UnionTextDecConfig
from .seq_dec_tf import *
from .text import *


@dataclass
class CNNTextSeqModelConfig(BaseConfig):
    seq_dec_conf: SeqDecoderVarySizeConfig
    text_dec_conf: UnionTextDecConfig
    backbone: str
    n_att: int
    n_in: int = 3
    trans_type: str = 'conv1'
    tf_n_head: int = None
    tf_n_ff: int = None
    tf_n_layer: int = None
    tf_dropout: float = 0
    weights: str = 'imagenet'

    @property
    def name(self):
        name = []
        tmp = f'{self.backbone}-{self.weights}-att{self.n_att}'
        if self.trans_type == 'conv1':
            pass
        elif self.trans_type == 'transformer':
            tmp += f'-tf({self.tf_n_head},{self.tf_n_ff})x{self.tf_n_layer}'
            if self.tf_dropout > 0:
                tmp += f'-drop{self.tf_dropout}'
        else:
            raise NotImplementedError()
        name.append(tmp)
        name.append(self.seq_dec_conf.name)
        name.append(self.text_dec_conf.name)
        return '/'.join(name)

    def make_model(self):
        return CNNTextSeqModel(self)


class CNNTextSeqModel(CNNTextSetModel):
    def __init__(self, conf: CNNTextSeqModelConfig):
        super(CNNTextSetModel, self).__init__()
        self.conf = conf
        self.backbone = get_encoder(conf.backbone,
                                    in_channels=conf.n_in,
                                    weights=conf.weights)
        self.trans1 = nn.Conv2d(self.backbone.out_channels[-1],
                                conf.n_att,
                                kernel_size=1)
        if conf.trans_type == 'transformer':
            self.pos = SinPositionEmbedding2D(num_pos_feats=self.conf.n_att //
                                              2)
            self.trans2 = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(self.conf.n_att,
                                           self.conf.tf_n_head,
                                           self.conf.tf_n_ff,
                                           self.conf.tf_dropout),
                num_layers=self.conf.tf_n_layer,
            )
        self.gen = conf.seq_dec_conf.make_model()
        self.tgt_dec = conf.text_dec_conf.make_model()

    def dec_parameters(self):
        return chain(self.gen.parameters(), self.tgt_dec.parameters())

    def forward(self,
                img,
                input_ids=None,
                lengths=None,
                labels=None,
                is_pad=None,
                **kwargs):
        """
        Args:
            img: (bs, c, h, w)
            input_ids: (n, t)
            lengths: (bs, )
            labels: (n, t)
        """
        ctx = self.forward_img(img)

        # prediction
        seq_dec = self.gen.forward(context_series=ctx, lengths=lengths)
        # (n, h*w, c)
        ctx_exp = expand_by_lengths(ctx, seq_dec.lengths)
        tgt_dec = self.tgt_dec.forward(input_ids=input_ids,
                                       text_vec=seq_dec.R,
                                       context_series=ctx_exp,
                                       labels=labels,
                                       is_empty=is_pad)

        loss = None
        if input_ids is not None:
            loss = tgt_dec.loss

        return Return(
            loss=loss,
            loss_tgt=tgt_dec.loss,
            pred_lengths=seq_dec.pred_lengths,
            logits=tgt_dec.logits,
            mems=tgt_dec.mems,
        )


@dataclass
class Return:
    loss: Tensor
    loss_tgt: Tensor
    pred_lengths: Tensor
    logits: Tensor
    mems: Tensor
