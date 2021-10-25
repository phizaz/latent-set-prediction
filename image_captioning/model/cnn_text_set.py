from itertools import chain

from segmentation_models_pytorch.encoders import get_encoder
from trainer.start import *

from .set_dec_tf import *
from .text import *

UnionTextEncConfig = Union[TextEncoderWithContextConfig]
UnionTextDecConfig = Union[TextPytorchDecoderConfig]
UnionSetDecConfig = Union[SetDecoderVarySizeConfig]


@dataclass
class CNNTextSetModelConfig(BaseConfig):
    set_dec_conf: UnionSetDecConfig
    text_enc_conf: UnionTextEncConfig
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
    dedup: bool = True

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
        name.append(self.set_dec_conf.name)
        name.append(self.text_enc_conf.name)
        name.append(self.text_dec_conf.name)
        return '/'.join(name)

    def make_model(self):
        return CNNTextSetModel(self)


class CNNTextSetModel(nn.Module):
    def __init__(self, conf: CNNTextSetModelConfig):
        super().__init__()
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
        self.gen = conf.set_dec_conf.make_model()
        self.tgt_enc = conf.text_enc_conf.make_model()
        self.tgt_dec = conf.text_dec_conf.make_model()

    def enc_parameters(self):
        enc = []
        enc.append(self.backbone.parameters())
        enc.append(self.trans1.parameters())
        if self.conf.trans_type == 'transformer':
            enc.append(self.trans2.parameters())
        return chain(*enc)

    def dec_parameters(self):
        return chain(self.gen.parameters(), self.tgt_enc.parameters(),
                     self.tgt_dec.parameters())

    def forward_img(self, img):
        ctx = self.backbone(img)[-1]
        ctx = self.trans1(ctx)
        bs, c, h, w = ctx.shape
        if self.conf.trans_type == 'transformer':
            # (bs, c, h, w)
            ctx = ctx + self.pos.query(h, w, img.device)
            # (bs, c, h*w)
            ctx = ctx.reshape(bs, c, h * w)
            # (h*w, bs, c)
            ctx = ctx.permute([2, 0, 1])
            # (h*w, bs, c)
            ctx = self.trans2.forward(ctx)
            # (bs, h*w, c)
            ctx = ctx.permute([1, 0, 2])
        elif self.conf.trans_type == 'conv1':
            # (bs, c, h*w)
            ctx = ctx.reshape(bs, c, h * w)
            # (bs, h*w, c)
            ctx = ctx.permute([0, 2, 1])
        else:
            raise NotImplementedError()
        return ctx

    def forward(self,
                img,
                input_ids=None,
                lengths=None,
                labels=None,
                index=None,
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
        # (n, h*w, c)
        ctx_exp = expand_by_lengths(ctx, lengths)
        if index is not None:
            index = expand_by_lengths(index, lengths)

        # target encoding
        if input_ids is not None:
            B = self.tgt_enc.forward(input_ids,
                                     context_series=ctx_exp,
                                     index=index)
        else:
            B = None

        # prediction
        set_dec = self.gen.forward(context_series=ctx, B=B, len_B=lengths)
        if type(self.gen) == SetDecoderVarySize:
            tgt_dec = self.tgt_dec.forward(input_ids=input_ids,
                                           text_vec=set_dec.R_pi,
                                           context_series=ctx_exp,
                                           labels=labels,
                                           is_empty=is_pad)
        else:
            raise NotImplementedError()

        loss = None
        if input_ids is not None:
            loss = tgt_dec.loss + set_dec.loss

        return Return(
            loss=loss,
            loss_tgt=tgt_dec.loss,
            loss_RB=set_dec.loss_RB,
            loss_BR=set_dec.loss_BR,
            loss_extra=set_dec.loss_extra,
            ordering=set_dec.R_i,
            pred_lengths=set_dec.pred_lengths,
            logits=tgt_dec.logits,
            mems=tgt_dec.mems,
        )

    def eval_forward(self, img) -> List[List[str]]:
        ctx = self.forward_img(img)

        # prediction
        seq_dec = self.gen.forward(context_series=ctx)
        ctx_exp = expand_by_lengths(ctx, seq_dec.pred_lengths)
        pred = self.tgt_dec.eval_forward(text_vec=seq_dec.R,
                                         context_series=ctx_exp)

        # output as strings
        tokenizer = self.conf.text_dec_conf.tokenizer
        strings = []
        for obj in pred:
            tokens = tokenizer.convert_ids_to_tokens(obj,
                                                     skip_special_tokens=True)
            strings.append(tokenizer.convert_tokens_to_string(tokens))

        # list of list of strings
        out = []
        offset = 0
        for length in seq_dec.pred_lengths:
            out.append(strings[offset:offset + length])
            offset += length
        return out


@dataclass
class Return:
    loss: Tensor = None
    loss_tgt: Tensor = None
    loss_RB: Tensor = None
    loss_BR: Tensor = None
    loss_extra: Tensor = None
    pred_lengths: Tensor = None
    ordering: Tensor = None
    logits: Tensor = None
    mems: Tensor = None
