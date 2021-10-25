from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class BoxEncoderV2Config:
    n_cls: int
    n_hid: int
    n_layer: int = 3
    n_common_layer: int = 1
    n_out: int = None
    dropout: float = 0
    ignore_cls_id: int = -1
    do_ignore: bool = False
    use_context: bool = True
    detach_context: bool = True
    context_type: str = 'avgr'
    mlp_pre_reduce: bool = True
    mlp_post_reduce: bool = True

    def make_model(self):
        return BoxEncoderV2(self)


class BoxEncoderV2(nn.Module):
    def __init__(self, conf: BoxEncoderV2Config):
        super().__init__()
        self.conf = conf
        if conf.n_out is None:
            conf.n_out = conf.n_hid
        self.cls_emb = nn.Embedding(conf.n_cls, conf.n_hid)
        self.cls_layers = MLPLNReLU(conf.n_hid,
                                    conf.n_layer,
                                    dropout=conf.dropout)
        self.xy_emb = nn.Linear(4, conf.n_hid)
        self.xy_layers = MLPLNReLU(conf.n_hid,
                                   conf.n_layer,
                                   dropout=conf.dropout)
        if conf.context_type == 'context':
            n_inp = 2048
        elif conf.context_type in ['avgr', 'maxr', 'memory']:
            n_inp = conf.n_hid
        else:
            raise NotImplementedError()
        self.ctx_layers = MLPLNReLU(conf.n_hid,
                                    conf.n_layer,
                                    dropout=self.conf.dropout,
                                    n_inp=n_inp)
        if conf.mlp_post_reduce:
            self.post_ctx_layers = MLPLNReLU(conf.n_hid,
                                             conf.n_layer,
                                             dropout=self.conf.dropout)
        self.output = MLPLNReLU(conf.n_hid * 3,
                                conf.n_common_layer,
                                last_relu=False,
                                n_out=conf.n_hid)

    def forward(self,
                xywh,
                cls_id,
                lengths,
                context=None,
                memory=None,
                context_R=None,
                context_R_lengths=None,
                **kwargs):
        """
        Args:
            xyxy: (n, 4)
            cls_id: (n)
            lengths: (bs, )
            context_R: (n, hid)
            context_R_lengths: (bs, )

        Returns: (n, n_hid)
        """
        a = self.cls_layers(self.cls_emb(cls_id))
        b = self.xy_layers(self.xy_emb(xywh))
        if self.conf.do_ignore:
            b[cls_id == self.conf.ignore_cls_id] = 0

        # (n, n_hid)
        if self.conf.use_context:
            if self.conf.context_type in ['avgr', 'maxr']:
                # using the average of r as the context
                context = context_R
                if self.conf.detach_context:
                    context = context.detach()
                if self.conf.mlp_pre_reduce:
                    # (n, hid)
                    context = self.ctx_layers.forward(context)
                # reduce
                # (bs, hid)
                if self.conf.context_type == 'avgr':
                    context = mean_vary_lengths(context, context_R_lengths)
                elif self.conf.context_type == 'maxr':
                    context = max_vary_lengths(context, context_R_lengths)
                else:
                    raise NotImplementedError()
                if self.conf.mlp_post_reduce:
                    context = self.post_ctx_layers.forward(context)
            elif self.conf.context_type == 'context':
                if self.conf.detach_context:
                    context = context.detach()
                context = self.ctx_layers.forward(context)
            elif self.conf.context_type == 'memory':
                context = memory
                if self.conf.detach_context:
                    context = context.detach()
                context = self.ctx_layers.forward(context)
            else:
                raise NotImplementedError()
            c = expand_by_lengths(context, lengths)
        else:
            c = torch.zeros_like(a)

        x = torch.cat([a, b, c], dim=-1)
        x = self.output(x)
        return x


@dataclass
class BoxEncoderV3Config:
    n_cls: int
    n_hid: int
    n_layer: int = 3
    n_common_layer: int = 1
    dropout: float = 0
    use_context: bool = True
    detach_context: bool = True
    context_type: str = 'avgr'
    mlp_pre_reduce: bool = True
    mlp_post_reduce: bool = True

    def make_model(self):
        return BoxEncoderV3(self)


class BoxEncoderV3(nn.Module):
    def __init__(self, conf: BoxEncoderV3Config):
        super().__init__()
        self.conf = conf
        self.cls_emb = nn.Embedding(conf.n_cls, conf.n_hid // 2)
        self.xy_emb = nn.Linear(4, conf.n_hid // 2)
        self.dropout = nn.Dropout(p=conf.dropout)
        self.layers = MLPLNReLU(conf.n_hid,
                                conf.n_layer,
                                dropout=self.conf.dropout)

        if conf.context_type == 'context':
            n_inp = 2048
        elif conf.context_type in ['avgr', 'maxr', 'memory']:
            n_inp = conf.n_hid
        else:
            raise NotImplementedError()
        self.ctx_layers = MLPLNReLU(conf.n_hid,
                                    conf.n_layer,
                                    dropout=self.conf.dropout,
                                    n_inp=n_inp)
        if conf.mlp_post_reduce:
            self.post_ctx_layers = MLPLNReLU(conf.n_hid,
                                             conf.n_layer,
                                             dropout=self.conf.dropout)
        self.output = MLPLNReLU(conf.n_hid * 2,
                                conf.n_common_layer,
                                n_out=conf.n_hid,
                                last_relu=False,
                                last_drop=False,
                                dropout=self.conf.dropout)

    def forward(self,
                xywh,
                cls_id,
                lengths=None,
                context=None,
                memory=None,
                context_R=None,
                context_R_lengths=None,
                **kwargs):
        """
        Args:
            xyxy: (n, 4)
            cls_id: (n)
            lengths: (bs, )
            context: (bs, 2048)
            memory: (bs, hid)
            index: (bs,)

        Returns: (n, n_hid)
        """
        a = self.layers.forward(
            torch.cat(
                [
                    self.dropout(self.cls_emb(cls_id)),
                    self.dropout(self.xy_emb(xywh)),
                ],
                dim=-1,
            ))

        # (n, n_hid)
        if self.conf.use_context:
            if self.conf.context_type in ['avgr', 'maxr']:
                # using the average of r as the context
                context = context_R
                if self.conf.detach_context:
                    context = context.detach()
                if self.conf.mlp_pre_reduce:
                    # (n, hid)
                    context = self.ctx_layers.forward(context)
                # reduce
                # (bs, hid)
                if self.conf.context_type == 'avgr':
                    context = mean_vary_lengths(context, context_R_lengths)
                elif self.conf.context_type == 'maxr':
                    context = max_vary_lengths(context, context_R_lengths)
                else:
                    raise NotImplementedError()
                if self.conf.mlp_post_reduce:
                    context = self.post_ctx_layers.forward(context)
            elif self.conf.context_type == 'context':
                if self.conf.detach_context:
                    context = context.detach()
                context = self.ctx_layers.forward(context)
            elif self.conf.context_type == 'memory':
                context = memory
                if self.conf.detach_context:
                    context = context.detach()
                context = self.ctx_layers.forward(context)
            else:
                raise NotImplementedError()
            c = expand_by_lengths(context, lengths)
        else:
            c = torch.zeros(len(a), self.conf.n_hid).to(a.device)

        x = torch.cat([a, c], dim=-1)
        x = self.output(x)
        return x


@dataclass
class BoxEncoderRepeatConfig:
    n_repeat: int
    enc_config: BoxEncoderV3Config

    def make_model(self):
        return BoxEncoderRepeat(self)


class BoxEncoderRepeat(nn.Module):
    def __init__(self, conf: BoxEncoderRepeatConfig):
        super().__init__()
        self.conf = conf
        self.enc = nn.ModuleList(
            [conf.enc_config.make_model() for i in range(conf.n_repeat)])

    def forward(self,
                xywh,
                cls_id,
                lengths=None,
                context=None,
                context_R=None,
                context_R_lengths=None,
                **kwargs):
        """
        Args:
            xyxy: (n, 4)
            cls_id: (n)
            lengths: (bs, )
            context_R: (n, hid * repeat)
            context_R_lengths: (bs, )
            index: (bs,)

        Returns: (n, n_hid)
        """
        dev = xywh.device
        # spilts
        Context_R = []
        n_hid = self.conf.enc_config.n_hid
        for i in range(self.conf.n_repeat):
            # (n, hid)
            Context_R.append(context_R[:, i * n_hid:(i + 1) * n_hid])

        # (n, hid * repeat)
        out = torch.zeros(len(xywh), n_hid * self.conf.n_repeat).to(dev)
        for i in range(self.conf.n_repeat):
            # (n, hid)
            out[:, i * n_hid:(i + 1) * n_hid] = self.enc[i].forward(
                xywh=xywh,
                cls_id=cls_id,
                lengths=lengths,
                context=context,
                context_R=Context_R[i],
                context_R_lengths=context_R_lengths)

        return out


class MLPLNReLU(nn.Sequential):
    def __init__(self,
                 n_hid: int,
                 n_layer: int,
                 last_relu: bool = True,
                 last_norm: bool = True,
                 last_drop: bool = True,
                 n_inp: int = None,
                 n_out: int = None,
                 dropout: float = 0):
        if n_inp is None:
            n_inp = n_hid
        if n_out is None:
            n_out = n_hid

        layers = []
        for i in range(n_layer):
            if i == 0 and i == n_layer - 1:
                layers.append(nn.Linear(n_inp, n_out))
                n = n_out
            elif i == 0:
                layers.append(nn.Linear(n_inp, n_hid))
                n = n_hid
            elif i == n_layer - 1:
                layers.append(nn.Linear(n_hid, n_out))
                n = n_out
            else:
                layers.append(nn.Linear(n_hid, n_hid))
                n = n_hid

            if i != n_layer - 1 or last_norm:
                layers.append(nn.LayerNorm(n))
            if i != n_layer - 1 or last_relu:
                layers.append(nn.ReLU(inplace=False))
            if i != n_layer - 1 or last_drop:
                layers.append(nn.Dropout(dropout))

        if len(layers) == 0:
            layers.append(nn.Identity())

        super().__init__(*layers)


def chunk_pad_by_lengths(x, lengths, batch_first: bool = False):
    """
    Args: 
        x: (n*t, d)
    Returns: 
        (t, n, d) if not batch_first
        (n, t, d) if batch_first
    """
    x = x.split(list(lengths), 0)
    x = nn.utils.rnn.pad_sequence(x, batch_first=batch_first)
    return x


def make_pad_mask(lengths, batch_first, max_t=None):
    """
    square matrix with row vectors of [False] * length[i]
    """
    dev = lengths.device
    bs = lengths.shape[0]
    if max_t is None:
        max_t = lengths.max()
    pad_mask = torch.arange(0, max_t).expand(bs, max_t).to(dev)
    pad_mask = pad_mask >= lengths.unsqueeze(-1)
    if not batch_first:
        pad_mask = pad_mask.permute([1, 0])
    return pad_mask


def expand_by_lengths(x, lengths):
    """
    Args:
        x: (bs, d)
        lengths: (bs, )
    Returns: (n*t, d)
    """
    # (n*t, )
    idx = lengths_to_idx(lengths)
    # (n*t, d)
    out = x[idx]
    return out


def lengths_to_idx(lengths: Tensor):
    """
    [1, 2] into [0, 1, 1]
    """
    idx = []
    for i, length in enumerate(lengths):
        idx += [i] * length
    return torch.LongTensor(idx).to(lengths.device)


def mean_vary_lengths(x, lengths):
    """
    Args:
        x: (n, hid)
        lengths: (bs, )
    """
    # (bs, t, hid)
    x = chunk_pad_by_lengths(x, lengths, batch_first=True)
    # (bs, t)
    pad = make_pad_mask(lengths, batch_first=True)
    x[pad] = 0
    # (bs, hid)
    dev = x.device
    # (bs, hid)
    x = x.sum(dim=1) / lengths.to(dev).unsqueeze(-1)
    return x


def max_vary_lengths(x, lengths):
    """
    Args:
        x: (n, hid)
        lengths: (bs, )
    """
    # (bs, t, hid)
    x = chunk_pad_by_lengths(x, lengths, batch_first=True)
    # (bs, t)
    pad = make_pad_mask(lengths, batch_first=True)
    x[pad] = float('-inf')
    # (bs, hid)
    x, _ = x.max(dim=1)
    return x
