import math

from trainer.start import *


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: (bs, t)
        """
        dev = x.device
        bs, max_t, _ = x.shape
        pos = torch.arange(max_t).unsqueeze(0).expand(bs, max_t).to(dev)
        pos = self.emb(pos)
        return x + pos

    def query(self, ids):
        pos = self.emb(ids)
        return pos


class SinPositionalEmbedding(nn.Module):
    """
    taken from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        # (t, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(max_len, d_model)
        self.emb.weight.data = pe
        self.emb.requires_grad_(False)

    def forward(self, x):
        """
        Args:
            x: (bs, t)
        """
        dev = x.device
        bs, max_t, _ = x.shape
        pos = torch.arange(max_t).unsqueeze(0).expand(bs, max_t).to(dev)
        pos = self.emb(pos)
        return x + pos

    def query(self, ids):
        pos = self.emb(ids)
        return pos


class SinPositionEmbedding2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def query(self, h, w, dev):
        not_mask = torch.ones(1, h, w).to(dev)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32,
                             device=dev)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos