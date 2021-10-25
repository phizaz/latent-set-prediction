import math

from trainer.start import *


class GaussianSeeder(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim
        self.mu = nn.Parameter(torch.zeros(n_dim))
        self.std = nn.Parameter(torch.ones(n_dim))

    def sample(self, bs, n):
        dev = self.mu.device
        out = torch.randn(bs, n, self.n_dim, device=dev) * self.std + self.mu
        return out


class LearnableSeeder(nn.Module):
    def __init__(self, n_dim, n_max_sentence):
        super().__init__()
        self.emb = nn.Embedding(n_max_sentence, n_dim)

    def sample(self, bs, n):
        dev = self.emb.weight.device
        idx = torch.arange(n).expand(bs, n).to(dev)
        out = self.emb(idx)
        return out


class SinSeeder(nn.Module):
    def __init__(self, n_dim, n_max_sentence):
        super().__init__()

        self.n_dim = n_dim
        pe = torch.zeros(n_max_sentence, n_dim)
        position = torch.arange(0, n_max_sentence,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_dim, 2).float() * (-math.log(10000.0) / n_dim))
        # (t, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, t, dim)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def sample(self, bs, n):
        """
        Returns: (bs, n, dim)
        """
        # (bs, n, dim)
        out = self.pe[:, :n, :].expand(bs, n, self.n_dim)
        return out
