from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


@dataclass
class LatentLossReturn:
    # ordered set elements (by hungarian with the ground truths)
    S_pi: Tensor
    # ordering index (S[S_i] == S_pi)
    S_i: Tensor
    # the latent loss
    loss: Tensor
    loss_sg: Tensor
    loss_gs: Tensor


class LSPLoss:
    """
    latent loss of the latent set prediction method

    Args:
        gcr_mode: 'gcr', 'gc'
        w_loss_gs: loss strength for the ground truth to move towards set elements 
        w_loss_sg: loss strength for the set elements to move towards the ground truths 
        d: (for gcr only) relative distance coefficient to omit gradient rejection
    """
    def __init__(self,
                 gcr_mode: str = 'gcr',
                 w_loss_gs: float = 1,
                 w_loss_sg: float = 0.1,
                 d: float = 1e-3):
        self.gcr_mode = gcr_mode
        self.w_loss_gs = w_loss_gs
        self.w_loss_sg = w_loss_sg
        self.d = d

    def forward(self, S, len_S, G, len_G):
        """
        Args:
            S: (n, c) n = total elements in a batch
            len_S: (n, ) specify the cardinality of each set in a batch
            G: (n, c) n = total elements in a batch (must be of the same size as S)
            len_G: (n, ) specify the cardinality of each set in a batch
        """
        func = batch_lsp(gcr_mode=self.gcr_mode, d=self.d)
        S_pi, S, G, S_i = func(S, len_S, G, len_G)
        # R => B must not push gradient to B
        loss_sg = F.mse_loss(G[S_i], S.detach(), reduction='none')
        loss_gs = F.mse_loss(S, S_pi.detach(), reduction='none')
        loss_sg = loss_sg.mean()
        loss_gs = loss_gs.mean()

        loss = self.w_loss_sg * loss_sg + self.w_loss_gs * loss_gs

        return LatentLossReturn(
            S_pi=S_pi,
            S_i=S_i,
            loss=loss,
            loss_gs=loss_gs,
            loss_sg=loss_sg,
        )


def batch_lsp(
    gcr_mode: str = 'gcr',
    d: float = 0,
):
    """
    Args:
        gc_mode: 'none', 'gc', 'gcr'
    """
    class Fn(torch.autograd.Function):
        """reject the gradinet direction that are not decreasing the distance."""
        @staticmethod
        def forward(ctx, G: Tensor, len_G: List[int], S: Tensor,
                    len_S: List[int]):
            """
            Args:
                always expect (n, c) vectors so that it's storage efficient.
            """
            if isinstance(len_G, Tensor):
                len_G = len_G.cpu().numpy()
            if isinstance(len_S, Tensor):
                len_S = len_S.cpu().numpy()
            assert all(
                n_gt <= n_pred for n_gt, n_pred in zip(len_G, len_S)
            ), f'there must be more predictions than the ground truths'

            # calculate the all-pair distances
            with torch.no_grad():
                # (bs, t1, t2)
                dists = cdist_vary_lengths(G, len_G, S, len_S).cpu()
                # cross-out
                for i, (n_a, n_b) in enumerate(zip(len_G, len_S)):
                    assert n_a > 0 and n_b > 0
                    dists[i, n_a:, :] = float('inf')
                    dists[i, :, n_b:] = float('inf')

            s_offset = 0
            g_offset = 0
            cols = []
            for j, (dist, n_gt, n_s) in enumerate(zip(dists, len_G, len_S)):
                cost = dist[:n_gt, :n_s].numpy()
                # NOTE: for debugging purposes
                if np.any(np.isnan(cost)):
                    print('cost:', cost)
                    print('len:', len_G, len_S)
                    print('pred:', S[s_offset:s_offset + 1])
                    print('gt:', G[g_offset:g_offset + 1])
                    raise ValueError('cost matrix contains nan')

                # NOTE: usually a bottleneck on large sets
                row, col = linear_sum_assignment(cost)
                col = torch.LongTensor(col)
                cols.append(s_offset + col)

                s_offset += n_s
                g_offset += n_gt
            # (n,)
            cols = torch.cat(cols)

            ctx.save_for_backward(cols, S, G)
            """
            Returns:
                S[cols]: used for decoding via prediction head
                G, S: used for calculating latent loss
                cols: reported matched indexes
            """
            return S[cols], G, S, cols

        @staticmethod
        def backward(ctx, S_pi_task_grad, G_latent_grad, S_latent_grad, *args):
            """
            the ordering of arguments follows the "return" of the forward part (S[cols], G, S, ...)

            Args:
                S_pi_task_grad: gradient on S[cols]
                G_latent_grad: gradient on gt, resultant of latent loss
                S_latent_grad: gradient on pred (no cols), resultant of latent loss
            """
            cols, S, G = ctx.saved_tensors

            # init
            S_task_grad = torch.zeros_like(S)
            G_task_grad = torch.zeros_like(G)

            # unsort pred's grad
            S_task_grad[cols] = S_pi_task_grad

            if gcr_mode in ['gc', 'gcr']:
                # clone the pred's grad => gt
                G_task_grad = S_pi_task_grad.clone()

            if gcr_mode in ['gcr']:
                # gradient rejection
                # using both its own latent grad + opposite latent grad to estimate the distance
                # we need to use "negative" of the opposite gradient
                G_inv_latent_grad = torch.zeros_like(S_latent_grad)
                G_inv_latent_grad[cols] = G_latent_grad
                S_task_grad = vector_reject_if_obtuse_with_safe_radius(
                    S_task_grad, S_latent_grad - G_inv_latent_grad, d)
                G_task_grad = vector_reject_if_obtuse_with_safe_radius(
                    G_task_grad, G_latent_grad - S_latent_grad[cols], d)

            # combine with latent grads
            S_grad = S_task_grad + S_latent_grad
            G_grad = G_task_grad + G_latent_grad
            return (G_grad, None, S_grad, None, None, None)

    return Fn.apply


def vector_reject_if_obtuse_with_safe_radius(vec, reject_vec, d: float):
    """remove reject_vec direction from vec only if they form obtuse angles

    Args:
        vec: (n, hid)
        reject_vec: (n, hid)
    """
    norm_rej = F.normalize(reject_vec, dim=-1)
    prod = (vec * norm_rej).sum(dim=-1, keepdim=True)
    # if obtuse => reject the gradient
    proj_vec = norm_rej * torch.where(prod < 0, prod, torch.zeros_like(prod))

    # safe radius = norm of the vec (aka. task gradient)
    safe_radius = vec.norm(dim=-1, keepdim=True) * d
    # if within safe radius, don't reject
    rej_norm = reject_vec.norm(dim=-1, keepdim=True)
    proj_vec = torch.where(rej_norm < safe_radius, torch.zeros_like(proj_vec),
                           proj_vec)
    vec = vec - proj_vec
    return vec


def chunk_pad_by_lengths(x: Tensor, lengths: Tensor, batch_first: bool):
    """
    (bs * k, c) => (bs, k, c) with proper zero paddings

    Args: 
        x: (n*t, d)
    Returns: 
        (t, n, d) if not batch_first
        (n, t, d) if batch_first
    """
    x = x.split(list(lengths), 0)
    x = nn.utils.rnn.pad_sequence(x, batch_first=batch_first)
    return x


def flat_by_lengths(x, lengths, batch_first):
    """
    Args:
        x: (n, t, c)
        batch_first: whether x is (n, t, c)
    Returns: (n*t, c)
    """
    assert x.dim() in (2, 3)
    if not batch_first:
        # t, n, c => n, t, c
        if x.dim() == 3:
            x = x.permute([1, 0, 2])
        elif x.dim() == 2:
            x = x.permute([1, 0])
        else:
            raise NotImplementedError()

    mask = ~make_pad_mask(lengths, batch_first=True)
    assert x.shape[:2] == mask.shape
    # (n, t, c) => (n*t, c)
    x = x.flatten(0, 1)
    # (n, t) => (n*t)
    mask = mask.flatten()
    out = x[mask].contiguous()
    return out


def cdist_vary_lengths(A, A_n, B, B_n, p=2):
    """
    Args:
        A: (n, dim)
        A_n: (bs)
        B: (m, dim)
        B_n: (bs)
    """
    # (bs, t1, dim)
    A = chunk_pad_by_lengths(A, A_n, batch_first=True).type(torch.float32)
    # (bs, t2, dim)
    B = chunk_pad_by_lengths(B, B_n, batch_first=True).type(torch.float32)
    # (bs, t1, t2)
    dists = torch.cdist(A, B, p=p)
    return dists


def cosine_vary_lengths(A, A_n, B, B_n):
    """
    Args:
        A: (n, dim)
        A_n: (bs)
        B: (m, dim)
        B_n: (bs)
    """
    # (bs, t1, dim)
    A = chunk_pad_by_lengths(A, A_n, batch_first=True).type(torch.float32)
    # (bs, t2, dim)
    B = chunk_pad_by_lengths(B, B_n, batch_first=True).type(torch.float32)
    # (bs, t1, t2)
    sim = torch.bmm(A, B.permute([0, 2, 1]))
    return sim


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
