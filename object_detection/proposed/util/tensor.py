import torch
from torch import nn


def chunk_by_lengths(x, lengths):
    """
    Args: 
        x: (n*t, d)
    """
    out = list(x.split(list(lengths), 0))
    return out


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
