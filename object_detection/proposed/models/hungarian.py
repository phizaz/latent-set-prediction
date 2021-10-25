from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


@dataclass
class MSEGCRLatentLossConfig:
    # latent loss gt => pred
    w_loss_br: float = 1
    # latent loss pred => gt
    w_loss_rb: float = 0.1
    # loss type
    loss: str = 'mse'
    # 'none', 'gc', 'gcr'
    gcr_mode: str = 'gcr'
    safe_coef: float = 0
    distance_both_ways: bool = False
    sum_over_n: bool = False

    def make_loss(self):
        return MSEGCRLatentLoss(self)


@dataclass
class LatentLossReturn:
    R_pi: Tensor
    R_i: Tensor
    loss: Tensor
    loss_RB: Tensor
    loss_BR: Tensor
    loss_extra: Tensor = None
    dists: Tensor = None


class MSEGCRLatentLoss:
    def __init__(self, conf: MSEGCRLatentLossConfig):
        self.conf = conf

    def forward(self, B, len_B, R, len_R, cols=None, dists_bias=None):
        """
        Args:
            cols: if present, enforce the assignment
        """
        func = batch_hungarian_gcr(
            gcr_mode=self.conf.gcr_mode,
            safe_coef=self.conf.safe_coef,
            distance_both_ways=self.conf.distance_both_ways)
        R_pi, B, R, R_i, dists = func(B, len_B, R, len_R, cols, dists_bias)

        loss_options = {
            'mse': F.mse_loss,
            'l1': F.l1_loss,
            'huber': F.smooth_l1_loss,
        }
        loss_fn = loss_options[self.conf.loss]

        # R => B must not push gradient to B
        # (n, hid)
        loss_RB = loss_fn(R[R_i], B.detach(), reduction='none')
        # (n, hid)
        loss_BR = loss_fn(B, R_pi.detach(), reduction='none')

        if self.conf.sum_over_n:
            # (bs, k, hid)
            loss_BR = chunk_pad_by_lengths(loss_BR, len_B, batch_first=True)
            # (bs, k, hid)
            loss_RB = chunk_pad_by_lengths(loss_RB, len_B, batch_first=True)
            loss_BR = loss_BR.sum(dim=1).mean()
            loss_RB = loss_RB.sum(dim=1).mean()
        else:
            loss_RB = loss_RB.mean()
            loss_BR = loss_BR.mean()

        loss = self.conf.w_loss_rb * loss_RB + self.conf.w_loss_br * loss_BR

        return LatentLossReturn(
            R_pi=R_pi,
            R_i=R_i,
            loss=loss,
            loss_BR=loss_BR,
            loss_RB=loss_RB,
            dists=dists,
        )


def batch_hungarian_gcr(
    gcr_mode: str = 'gcr',
    safe_coef: float = 0,
    distance_both_ways: bool = False,
):
    """
    Args:
        gc_mode: 'none', 'gc', 'gcr'
        map_extra: should we also map the unmatched to the closest?
    """
    class Fn(torch.autograd.Function):
        """reject the gradinet direction that are not decreasing the distance."""
        @staticmethod
        def forward(ctx,
                    GT: Tensor,
                    len_GT: List[int],
                    Pred: Tensor,
                    len_pred: List[int],
                    cols: Tensor = None,
                    dists_bias: Tensor = None):
            """
            Args:
                cols: if present, doesn't do Hungarian
            """
            if isinstance(len_GT, Tensor):
                len_GT = len_GT.cpu().numpy()
            if isinstance(len_pred, Tensor):
                len_pred = len_pred.cpu().numpy()
            assert all(
                n_gt <= n_pred for n_gt, n_pred in zip(len_GT, len_pred)
            ), f'there must be more predictions than the ground truths'

            if cols is None:
                with torch.no_grad():
                    # (bs, t1, t2)
                    dists = cdist_vary_lengths(GT, len_GT, Pred, len_pred)
                    if dists_bias is not None:
                        dists += dists_bias
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
                    cost = dist[:n_gt, :n_pred].cpu().numpy()
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
            return Pred[cols], GT, Pred, cols, dists

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
                    inv_gt_latent_grad = torch.zeros_like(gt_latent_grad)
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
            return (gt_grad, None, pred_grad, None, None, None, None, None)

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
