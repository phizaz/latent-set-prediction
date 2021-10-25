from trainer.start import *
from utils.assignment import *
from utils.utils import *

from model.set_dec_tf import SetDecoderVarySize
from .seeder import *


@dataclass
class SeqDecoderVarySizeConfig(BaseConfig):
    n_max_items: int
    n_hid: int
    context_type: str = 'series'
    dict_type: str = 'sin'
    sep_num_head: bool = False
    n_num_element_layer: int = 1
    n_head: int = 4
    n_ff: int = 512
    n_layer: int = 3
    dropout: float = 0
    use_pad_mask: bool = True
    causal_mask: bool = False
    w_loss_num: float = 1

    @property
    def name(self):
        name = f'seqdecvary({self.n_hid},{self.n_head},{self.n_ff})x{self.n_layer}'
        if self.dropout > 0:
            name += f'-drop{self.dropout}'
        name += f'-max{self.n_max_items}'
        name += f'-dict{self.dict_type}-elem{self.n_num_element_layer}'
        if self.use_pad_mask:
            name += f'-pad'
        if self.causal_mask:
            name += f'-causal'
        if self.context_type != 'series':
            name += f'-{self.context_type}'
        return name

    def make_model(self):
        return SeqDecoderVarySize(self)


class SeqDecoderVarySize(SetDecoderVarySize):
    def __init__(self, conf: SeqDecoderVarySizeConfig):
        super(SetDecoderVarySize, self).__init__()
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

    def forward(
        self,
        context=None,
        context_series=None,
        lengths=None,
    ):
        """
        Args:
            context: (bs, t, c)
        """
        with time_elapsed_to_profiler('generate/items'):
            gen = self.gen_items(context=context,
                                 context_series=context_series,
                                 lengths=lengths)
        return gen
