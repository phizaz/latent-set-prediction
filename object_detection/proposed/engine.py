# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
from models.detr import DETR, SetCriterion
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

import time
from contextlib import contextmanager


@contextmanager
def time_elapsed(key, prefix='profiler/', verbose=False):
    """a contextmanager that logs the time of the block
    Args:
        i_itr: only required once per iteration
    """
    begin_time = time.time()
    yield
    end_time = time.time()
    elapsed = end_time - begin_time
    if verbose: print(f'{prefix}{key}:', elapsed)


def train_one_epoch(model: DETR,
                    criterion: SetCriterion,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 11

    for samples, targets, index in metric_logger.log_every(
            data_loader, print_freq, header):
        with time_elapsed('iteration', verbose=True):
            samples = samples.to(device)
            targets = [{k: v.to(device)
                        for k, v in t.items()} for t in targets]

            with time_elapsed('model'):
                outputs = model(samples, targets=targets)

            # print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)
            # print(targets)
            with time_elapsed('criterion'):
                loss_dict, _, extra_dict = criterion(
                    outputs,
                    targets=targets,
                    index=index,
                    class_embed=model.class_embed,
                    bbox_embed=model.bbox_embed)
            weight_dict = criterion.weight_dict

            # move loss set to the detr's build function

            losses = sum(loss_dict[k] * weight_dict[k]
                         for k in loss_dict.keys() if k in weight_dict)
            # print({k : loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict})
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {
                f'{k}_unscaled': v
                for k, v in loss_dict_reduced.items()
            }
            loss_dict_reduced_scaled = {
                k: v * weight_dict[k]
                for k, v in loss_dict_reduced.items() if k in weight_dict
            }
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            with time_elapsed('optimize'):
                optimizer.zero_grad()
                with time_elapsed('optimize/backward', verbose=False):
                    losses.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_norm)
                with time_elapsed('optimize/step'):
                    optimizer.step()

            metric_logger.update(loss=loss_value,
                                 **loss_dict_reduced_scaled,
                                 **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # add more meters!
            metric_logger.update(**extra_dict)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: DETR, criterion: SetCriterion, postprocessors, data_loader,
             base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox')
                      if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets, index in metric_logger.log_every(
            data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, targets=targets)
        loss_dict, pred, extra_dict = criterion(outputs,
                                                targets=targets,
                                                index=index,
                                                class_embed=model.class_embed,
                                                bbox_embed=model.bbox_embed)
        weight_dict = criterion.weight_dict

        # outputs['pred_logits'] = pred[0]
        # outputs['pred_boxes'] = pred[1]

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        # extra metric
        metric_logger.update(**extra_dict)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets],
                                        dim=0)
        # print()
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs,
                                             orig_target_sizes, target_sizes)
        res = {
            target['image_id'].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes,
                                                  orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval[
                'bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval[
                'segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
