from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
import torch.nn.functional as F


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


def batch_hungarian_gcr(
    gcr_mode: str = 'gcr',
    safe_coef: float = 0,
    distance_both_ways: bool = True,
):
    """
    Args:
        gc_mode: 'none', 'gc', 'gcr'
        map_extra: should we also map the unmatched to the closest?
    """
    class Fn(torch.autograd.Function):
        """reject the gradinet direction that are not decreasing the distance."""
        @staticmethod
        def forward(ctx, GT: Tensor, len_GT: List[int], Pred: Tensor,
                    len_pred: List[int]):
            if isinstance(len_GT, Tensor):
                len_GT = len_GT.cpu().numpy()
            if isinstance(len_pred, Tensor):
                len_pred = len_pred.cpu().numpy()
            assert all(
                n_gt <= n_pred for n_gt, n_pred in zip(len_GT, len_pred)
            ), f'there must be more predictions than the ground truths'

            with torch.no_grad():
                # (bs, t1, t2)
                dists = cdist_vary_lengths(GT, len_GT, Pred, len_pred).cpu()
                # cross-out
                for i, (n_a, n_b) in enumerate(zip(len_GT, len_pred)):
                    assert n_a > 0 and n_b > 0
                    dists[i, n_a:, :] = float('inf')
                    dists[i, :, n_b:] = float('inf')

            pred_offset = 0
            gt_offset = 0
            cols = []
            for j, (dist, n_gt,
                    n_pred) in enumerate(zip(dists, len_GT, len_pred)):
                cost = dist[:n_gt, :n_pred].numpy()
                if np.any(np.isnan(cost)):
                    print('cost:', cost)
                    print('len:', len_GT, len_pred)
                    print('pred:', Pred[pred_offset:pred_offset + 1])
                    print('gt:', GT[gt_offset:gt_offset + 1])
                    raise ValueError('cost matrix contains nan')
                row, col = linear_sum_assignment(cost)
                col = torch.LongTensor(col)
                cols.append(pred_offset + col)

                pred_offset += n_pred
                gt_offset += n_gt
            # (n,)
            cols = torch.cat(cols)

            ctx.save_for_backward(cols, Pred, GT)
            """
            Returns:
                Pred[cols]: used for decoding via prediction head
                GT, Pred: used for latent loss
                cols: reported matched indexes
                unmatched_i: Pred's indexes that are unmatched (excess)
                unmatched_tgt: closest Pred's indexes to each of the unmatched
            """
            return Pred[cols], GT, Pred, cols

        @staticmethod
        def backward(ctx, pred_task_grad, gt_latent_grad, pred_latent_grad,
                     *args):
            """
            Args:
                pred_task_grad: gradient on pred[cols]
                gt_latent_grad: gradient on gt, resultant of latent loss
                pred_latent_grad: gradient on pred (no cols), resultant of latent loss
            """
            cols, Pred, GT = ctx.saved_tensors

            # init
            pred_task_grad_align = torch.zeros_like(Pred)
            gt_task_grad = torch.zeros_like(GT)

            # unsort pred's grad
            pred_task_grad_align[cols] = pred_task_grad

            if gcr_mode in ['gc', 'gcr']:
                # clone the pred's grad => gt
                gt_task_grad = pred_task_grad.clone()

            if gcr_mode in ['gcr']:
                # gradient rejection
                # print('pred_task_latent:', pred_task_grad_align.norm(dim=-1).median())
                # print('pred_latent:', pred_latent_grad.norm(dim=-1).median())
                if distance_both_ways:
                    # using both its own latent grad + opposite latent grad to estimate the distance
                    # this makes more sense!
                    # we need to use "negative" of the opposite gradient
                    inv_gt_latent_grad = torch.zeros_like(pred_latent_grad)
                    inv_gt_latent_grad[cols] = gt_latent_grad
                    pred_task_grad_align = vector_reject_if_obtuse_with_safe_radius(
                        pred_task_grad_align,
                        pred_latent_grad - inv_gt_latent_grad, safe_coef)
                    gt_task_grad = vector_reject_if_obtuse_with_safe_radius(
                        gt_task_grad, gt_latent_grad - pred_latent_grad[cols],
                        safe_coef)
                else:
                    # using its own latent gradient to calculate the distance
                    pred_task_grad_align = vector_reject_if_obtuse_with_safe_radius(
                        pred_task_grad_align, pred_latent_grad, safe_coef)
                    gt_task_grad = vector_reject_if_obtuse_with_safe_radius(
                        gt_task_grad, gt_latent_grad, safe_coef)

            # combine with latent grads
            pred_grad = pred_task_grad_align + pred_latent_grad
            gt_grad = gt_task_grad + gt_latent_grad
            return (gt_grad, None, pred_grad, None, None, None)

    return Fn.apply


def vector_reject_if_obtuse_with_safe_radius(vec, reject_vec,
                                             safe_coef: float):
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
    safe_radius = vec.norm(dim=-1, keepdim=True) * safe_coef
    # if within safe radius, don't reject
    rej_norm = reject_vec.norm(dim=-1, keepdim=True)
    proj_vec = torch.where(rej_norm < safe_radius, torch.zeros_like(proj_vec),
                           proj_vec)
    vec = vec - proj_vec
    return vec
