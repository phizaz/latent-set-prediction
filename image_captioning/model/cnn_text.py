from itertools import chain

from segmentation_models_pytorch.encoders import get_encoder
from trainer.start import *

from .cnn_text_set import CNNTextSetModel, UnionTextDecConfig
from .text import *


@dataclass
class CNNTextModelConfig(BaseConfig):
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
        tmp = f'cnntext-{self.backbone}-{self.weights}-att{self.n_att}'
        if self.trans_type == 'conv1':
            pass
        elif self.trans_type == 'transformer':
            tmp += f'-tf({self.tf_n_head},{self.tf_n_ff})x{self.tf_n_layer}'
            if self.tf_dropout > 0:
                tmp += f'-drop{self.tf_dropout}'
        else:
            raise NotImplementedError()
        name.append(tmp)
        name.append(self.text_dec_conf.name)
        return '/'.join(name)

    def make_model(self):
        return CNNTextModel(self)


class CNNTextModel(CNNTextSetModel):
    def __init__(self, conf: CNNTextModelConfig):
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
        self.dec = conf.text_dec_conf.make_model()

    def enc_parameters(self):
        enc = []
        enc.append(self.backbone.parameters())
        enc.append(self.trans1.parameters())
        if self.conf.trans_type == 'transformer':
            enc.append(self.trans2.parameters())
        return chain(*enc)

    def dec_parameters(self):
        return self.dec.parameters()

    def forward(self, img, input_ids=None, labels=None, **kwargs):
        """
        Args:
            img: (bs, c, h, w)
            input_ids: (n, t)
            labels: (n, t)
        """
        # (bs, h*w, c)
        ctx = self.forward_img(img)

        # prediction
        dec = self.dec.forward(input_ids=input_ids,
                               context_series=ctx,
                               labels=labels)

        loss = None
        if input_ids is not None:
            loss = dec.loss

        return Return(
            loss=loss,
            logits=dec.logits,
            mems=dec.mems,
        )

    def eval_forward(self, img) -> List[List[str]]:
        # (bs, h*w, c)
        ctx = self.forward_img(img)

        # prediction
        # (bs, t)
        pred = self.dec.eval_forward(context_series=ctx)

        tokenizer = self.conf.text_dec_conf.tokenizer
        # separated by <sep>
        # (bs, k, t)
        separated = []
        for ids in pred:
            instance = []
            obj = []
            for id in ids:
                if id == tokenizer.sep_token_id:
                    instance.append(obj)
                    obj = []
                else:
                    obj.append(id)
            instance.append(obj)
            separated.append(instance)

        # output as strings
        strings = []
        for instance in separated:
            tmp = []
            for obj in instance:
                tokens = tokenizer.convert_ids_to_tokens(
                    obj, skip_special_tokens=True)
                tmp.append(tokenizer.convert_tokens_to_string(tokens))
            strings.append(tmp)
        return strings


@dataclass
class Return:
    loss: Tensor
    logits: Tensor
    mems: Tensor
