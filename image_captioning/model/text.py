from trainer.start import *
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerFast)
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_outputs import dataclass as outputclass
from transformers.tokenization_utils import PreTrainedTokenizer
from utils.utils import *

from .positional_embedding import *


@dataclass
class TextEncoderWithContextConfig(BaseConfig):
    tokenizer: PreTrainedTokenizerFast
    n_max_length: int
    n_hid: int
    n_head: int
    n_ff: int
    n_layer: int
    dropout: float = 0
    pos_emb_type: str = 'sin'

    @property
    def name(self):
        name = f'encwctx({self.n_hid},{self.n_head},{self.n_ff})x{self.n_layer}'
        if self.dropout > 0:
            name += f'drop{self.dropout}'
        name += f'-pos{self.pos_emb_type}'
        return name

    def make_model(self):
        return TextEncoderWithContext(self)


class TextEncoderWithContext(nn.Module):
    def __init__(self, conf: TextEncoderWithContextConfig):
        super().__init__()
        self.conf = conf
        self.emb = nn.Embedding(len(conf.tokenizer), conf.n_hid)
        if conf.pos_emb_type == 'sin':
            self.pos = SinPositionalEmbedding(conf.n_hid, conf.n_max_length)
        elif conf.pos_emb_type == 'learn':
            self.pos = LearnablePositionalEmbedding(conf.n_hid,
                                                    conf.n_max_length)
        else:
            raise NotImplementedError()
        self.input_norm = nn.LayerNorm(conf.n_hid)

        self.encoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(conf.n_hid, conf.n_head, conf.n_ff,
                                       conf.dropout),
            num_layers=conf.n_layer,
        )

    def forward(self, input_ids, context_series=None, **kwargs):
        """
        Args:
            input_ids: (bs, length)
            context: (bs, k, hid)

        Returns: (bs, n_hid)
        """
        # (bs, length, n_hid)
        x = self.emb(input_ids)
        x = self.pos(x)
        x = self.input_norm(x)
        # (length, bs, n_hid)
        x = x.permute([1, 0, 2])

        pad_mask = input_ids == self.conf.tokenizer.pad_token_id
        # (k, bs, hid)
        context_series = context_series.permute([1, 0, 2])
        x = self.encoder.forward(x,
                                 memory=context_series,
                                 tgt_key_padding_mask=pad_mask)
        # (bs, length, n_hid)
        x = x.permute([1, 0, 2])

        # (bs, n_hid)
        head = x[:, 0, :]
        return head


@outputclass
class DecoderOutput(ModelOutput):
    loss: Tensor = None
    logits: Tensor = None
    mems: Tensor = None


@dataclass
class TextPytorchDecoderConfig(BaseConfig):
    tokenizer: PreTrainedTokenizer
    n_max_length: int
    n_hid: int
    n_head: int
    n_ff: int
    n_layer: int
    dropout: float = 0
    w_empty: float = None
    pos_emb_type: str = 'sin'
    loss_ignore_pad: bool = False
    loss_balance_eos: bool = False
    base_config = None
    use_cache: bool = False

    @property
    def name(self):
        name = f'decpt({self.n_hid},{self.n_head},{self.n_ff})x{self.n_layer}'
        if self.dropout > 0:
            name += f'drop{self.dropout}'
        name += f'-pos{self.pos_emb_type}'
        if self.w_empty is not None:
            name += f'-wempty{self.w_empty}'
        if self.loss_ignore_pad:
            name += f'-nopad'
        if self.loss_balance_eos:
            name += f'-baleos'
        if self.use_cache:
            name += f'-cache'
        return name

    def make_model(self):
        return TextPytorchDecoder(self)


class TextPytorchDecoder(PreTrainedModel):
    def __init__(self, conf: TextPytorchDecoderConfig):
        if conf.base_config is None:
            conf.base_config = PretrainedConfig()
        # required by the generator mixin
        conf.base_config.vocab_size = len(conf.tokenizer)
        super().__init__(conf.base_config)

        self.conf = conf
        self.emb = nn.Embedding(len(conf.tokenizer), conf.n_hid)
        if conf.pos_emb_type == 'sin':
            self.pos = SinPositionalEmbedding(conf.n_hid, conf.n_max_length)
        elif conf.pos_emb_type == 'learn':
            self.pos = LearnablePositionalEmbedding(conf.n_hid,
                                                    conf.n_max_length)
        else:
            raise NotImplementedError()
        self.input_norm = nn.LayerNorm(conf.n_hid)
        self.head = nn.Linear(conf.n_hid, len(conf.tokenizer))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(conf.n_hid, conf.n_head, conf.n_ff,
                                       conf.dropout),
            num_layers=conf.n_layer,
        )

    def get_output_embeddings(self):
        return self.head

    def loss(self, logits, labels, is_empty=None):
        """
        Args:
            logits: (n, t, out)
            labels: (n, t)
            is_empty: (n, )
        """
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # weights
            # (n, t)
            weights = torch.ones_like(shift_labels).float()
            if self.conf.loss_balance_eos:
                # there is one eos
                weights[shift_labels == self.conf.tokenizer.
                        eos_token_id] = weights.shape[1] - 1

            if self.conf.w_empty is not None:
                # (n, )
                w = torch.ones_like(is_empty).float()
                w[is_empty] = self.conf.w_empty
                weights *= w.unsqueeze(-1)

            # average of the whole batch
            weights = weights / weights.sum()

            # ignore pad labels
            if self.conf.loss_ignore_pad:
                shift_labels[shift_labels ==
                             self.conf.tokenizer.pad_token_id] = -100

            ce_loss = nn.CrossEntropyLoss(reduction='none')
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            # (n, t)
            loss = ce_loss(flat_logits, flat_labels)
            loss = (loss * weights.view(-1)).sum()
        return loss

    def forward(self,
                input_ids=None,
                text_vec=None,
                context_series=None,
                labels=None,
                is_empty=None,
                **kwargs):
        """
        Args:
            input_ids: teacher forcing signal (bs, t)
            sent_vec: (bs, hid)
            context: (bs, k, hid)
            mems: tuple (current decoded length, transformer state)
        """
        bs, max_t = input_ids.shape
        dev = input_ids.device

        # (bs, t, hid)
        x = self.emb(input_ids)
        # (n, t, n_hid)
        x = self.pos.forward(x)
        if text_vec is not None:
            # (bs, 1, hid)
            text_vec = text_vec.unsqueeze(1)
            x = x + text_vec
        x = self.input_norm(x)
        # (t, bs, hid)
        x = x.permute([1, 0, 2])

        # pad mask => (bs, max_t)
        pad_mask = input_ids == self.conf.tokenizer.pad_token_id
        # attention mask
        attn_mask = torch.triu(torch.ones(max_t, max_t).bool(),
                               diagonal=1).to(dev)

        # (t, bs, hid)
        context_series = context_series.permute([1, 0, 2])
        x = self.decoder.forward(tgt=x,
                                 memory=context_series,
                                 tgt_mask=attn_mask,
                                 tgt_key_padding_mask=pad_mask)
        # (bs, t, hid)
        x = x.permute([1, 0, 2])
        # (bs, t, word)
        logits = self.head(x)
        loss = self.loss(logits, labels, is_empty=is_empty)

        return DecoderOutput(
            loss=loss,
            logits=logits,
            mems=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # TODO: transformer cannot generate with cache
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            # print('past:', past[0].shape)
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "mems": past,
            "use_cache": kwargs.get("use_cache"),
            **kwargs,
        }

    def eval_forward(self, text_vec=None, context_series=None, **kwargs):
        # generate
        # no need to rearrange, we rearrange the loss function instead
        dev = context_series.device
        bs = len(context_series)
        # starts with <bos> (make sure that the dataset follows this)
        input_ids = torch.tensor([self.conf.tokenizer.bos_token_id] *
                                 bs).to(dev).unsqueeze(1)
        pred = self.generate(
            input_ids=input_ids,
            text_vec=text_vec,
            context_series=context_series,
            max_length=self.conf.n_max_length,
            use_cache=self.conf.use_cache,
            pad_token_id=self.conf.tokenizer.pad_token_id,
            eos_token_id=self.conf.tokenizer.eos_token_id,
        )
        return pred
