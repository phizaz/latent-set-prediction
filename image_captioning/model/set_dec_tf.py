from trainer.start import *
from utils.latent_loss import *
from utils.utils import *

from .seeder import *

UnionLatentLossConfig = Union[MSEGCRLatentLossConfig]


@dataclass
class SetDecoderVarySizeConfig(BaseConfig):
    n_max_items: int
    n_hid: int
    context_type: str = 'series'
    dict_type: str = 'sin'
    n_head: int = 4
    n_ff: int = 512
    n_layer: int = 3
    dropout: float = 0
    use_pad_mask: bool = True
    causal_mask: bool = False
    loss_conf: UnionLatentLossConfig = None

    @property
    def name(self):
        name = f'setdecvary({self.n_hid},{self.n_head},{self.n_ff})x{self.n_layer}'
        if self.dropout > 0:
            name += f'-drop{self.dropout}'
        name += f'-max{self.n_max_items}'
        name += f'-dict{self.dict_type}'
        if self.use_pad_mask:
            name += f'-pad'
        if self.causal_mask:
            name += f'-causal'
        name += f'-{self.loss_conf.name}'
        if self.context_type != 'series':
            name += f'-{self.context_type}'
        return name

    def make_model(self):
        return SetDecoderVarySize(self)


@dataclass
class GenReturn:
    R: Tensor = None
    lengths: Tensor = None
    pred_lengths: Tensor = None


class SetDecoderVarySize(nn.Module):
    def __init__(self, conf: SetDecoderVarySizeConfig):
        super().__init__()
        self.conf = conf

        if conf.dict_type == 'static':
            self.seeder = LearnableSeeder(conf.n_hid, conf.n_max_items)
        elif conf.dict_type == 'gaussian':
            self.seeder = GaussianSeeder(conf.n_hid)
        elif conf.dict_type == 'sin':
            self.seeder = SinSeeder(conf.n_hid, conf.n_max_items)
        else:
            raise NotImplementedError()

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=conf.n_hid,
                nhead=conf.n_head,
                dim_feedforward=conf.n_ff,
                dropout=conf.dropout,
            ),
            num_layers=conf.n_layer,
        )
        self.input_norm = nn.LayerNorm(conf.n_hid)
        self.latent_loss = self.conf.loss_conf.make_loss()

    def gen_items(
        self,
        context=None,
        context_series=None,
        lengths=None,
    ):
        """
        Args:
            context: (bs, t, c)
            lengths: (bs,)
        Returns:
            items: (n*t) if mlp
            items: (n*max_t) if each
            items: (n*t) if eachseq
        """
        if self.conf.context_type == 'vector':
            dev = context.device
            bs = context.shape[0]
        elif self.conf.context_type == 'series':
            dev = context_series.device
            bs = context_series.shape[0]
        else:
            raise NotImplementedError()
        is_supervised = lengths is not None

        if self.conf.context_type == 'series':
            # (t, bs, c)
            memory = context_series.permute([1, 0, 2])
        elif self.conf.context_type == 'vector':
            # (1, bs, c)
            memory = context.unsqueeze(0)

        # use the maximum length
        len_gen = len_pred = torch.LongTensor([self.conf.n_max_items] *
                                              bs).to(dev)
        # 'none' always generate full length
        # we treat it so even during train
        lengths = len_gen

        t = len_gen.max()
        # (bs, t, dim)
        R = self.seeder.sample(bs, t).type_as(memory)

        if self.conf.use_pad_mask:
            # (bs, t)
            pad_mask = make_pad_mask(len_gen, batch_first=True)
        else:
            pad_mask = None

        # attention mask
        if self.conf.causal_mask:
            # (t, t)
            attn_mask = torch.triu(torch.ones(t, t).bool(), diagonal=1).to(dev)
        else:
            attn_mask = None

        # (t, bs, dim)
        R = R.permute([1, 0, 2])
        R = self.input_norm(R)
        R = self.transformer.forward(R,
                                     memory,
                                     tgt_mask=attn_mask,
                                     tgt_key_padding_mask=pad_mask)
        # (bs, t, c)
        R = R.permute([1, 0, 2])

        # (n, c)
        R_flat = R.reshape(bs * t, -1)

        return GenReturn(
            lengths=lengths if is_supervised else len_pred,
            pred_lengths=len_pred,
            R=R_flat,
        )

    def hungarian_forward(self,
                          gen: GenReturn,
                          B: Tensor = None,
                          len_B: Tensor = None):
        R = gen.R

        len_R = gen.lengths
        pred_lengths = gen.pred_lengths

        if B is not None:
            # train
            with time_elapsed_to_profiler('generate/match'):
                loss_out = self.latent_loss.forward(B, len_B, R, len_R)

            loss = loss_out.loss
            R_pi = loss_out.R_pi
            return Return(
                R=R,
                B=B,
                R_pi=R_pi,
                R_i=loss_out.R_i,
                lengths=len_R,
                pred_lengths=pred_lengths,
                loss=loss,
                loss_RB=getattr(loss_out, 'loss_RB', None),
                loss_BR=getattr(loss_out, 'loss_BR', None),
                loss_extra=getattr(loss_out, 'loss_extra', None),
            )
        else:
            # inference
            # cannot do the mapping
            pred_lengths = gen.pred_lengths

            return Return(
                R=R,
                B=B,
                R_pi=None,
                lengths=len_R,
                pred_lengths=pred_lengths,
            )

    def forward(
        self,
        context=None,
        context_series=None,
        B=None,
        len_B=None,
    ):
        """
        Args:
            context: (bs, t, c)
            B: (n, n_hid) could be omitted during generation
            len_B: (bs,) could be supplied even during inference
        """

        with time_elapsed_to_profiler('generate/items'):
            gen = self.gen_items(context=context,
                                 context_series=context_series,
                                 lengths=len_B)

        return self.hungarian_forward(gen, B=B, len_B=len_B)


@dataclass
class Return:
    B: Tensor = None
    R: Tensor = None
    R_pi: Tensor = None
    R_i: Tensor = None
    lengths: Tensor = None
    pred_lengths: Tensor = None
    loss: Tensor = None
    loss_RB: Tensor = None
    loss_BR: Tensor = None
    loss_extra: Tensor = None
