# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

# debug
# a bit faster on my computer
torch.set_num_threads(1)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector',
                                     add_help=False)

    # track switches?
    parser.add_argument('--track_switches', default=1, type=int)
    # dslp at?
    # 'criterion', 'model'
    parser.add_argument('--dslp_at', default='criterion', type=str)
    # DETR's prediction head
    parser.add_argument('--decoder', default='v1', type=str)
    # wheter to use aux losses
    parser.add_argument('--use_aux', default=1, type=int)
    # aux strength
    parser.add_argument('--aux_coef', default=1, type=float)
    # aux set strength
    parser.add_argument('--aux_set_coef', default=1, type=float)
    # mode = 'none', 'gc', 'gcr'
    parser.add_argument('--mode', default='gcr', type=str)
    # 'all' = match all layers
    # 'top' = match only the top layers, but calculate losses for all layers still
    parser.add_argument('--match_mode', default='all', type=str)
    # whether to share encoder across layers
    # 'none', 'emb', 'emb+out', 'emb+ctx'
    parser.add_argument('--enc_share', default='none', type=str)
    # encoder type
    parser.add_argument('--enc', default='v1', type=str)
    # use context?
    parser.add_argument('--enc_context', default=1, type=int)
    # what context to use?
    # 'context' (for criterion's dslp), 'memory' (for model's dslp)
    parser.add_argument('--enc_context_type', default='context', type=str)
    parser.add_argument('--enc_mlp_pre_reduce', default=1, type=int)
    parser.add_argument('--enc_mlp_post_reduce', default=1, type=int)
    parser.add_argument('--enc_common_layer', default=1, type=int)

    # whether to use the top matching as another aux loss
    parser.add_argument('--aux_top_matching', default=0, type=int)
    parser.add_argument('--aux_top_coef', default=0.5, type=float)
    parser.add_argument('--majority_scaling', default=0, type=int)
    # whether to cumulate dists over the layers
    parser.add_argument('--cumulative_dists', default=0, type=int)

    parser.add_argument('--safe_coef', default=0.01, type=float)
    parser.add_argument('--sum_over_n', default=0, type=int)
    # 'gcr', 'gc' for aux layers
    parser.add_argument('--aux_mode', default=None, type=str)
    parser.add_argument('--aux_safe_coef', default=None, type=float)
    parser.add_argument('--distance_both_ways', default=1, type=int)
    parser.add_argument('--w_pred_gt', default=0.1, type=float)
    parser.add_argument('--w_gt_pred', default=1.0, type=float)
    parser.add_argument('--ignore_nocls_bbox', default=0, type=int)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=160, type=int)
    parser.add_argument('--lr_drop', default=120, type=int)
    parser.add_argument('--clip_max_norm',
                        default=0.1,
                        type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument(
        '--frozen_weights',
        type=str,
        default=None,
        help=
        "Path to the pretrained model. If set, only the mask head will be trained"
    )
    # * Backbone
    parser.add_argument('--backbone',
                        default='resnet50',
                        type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument(
        '--dilation',
        action='store_true',
        help=
        "If true, we replace stride with dilation in the last convolutional block (DC5)"
    )
    parser.add_argument(
        '--position_embedding',
        default='sine',
        type=str,
        choices=('sine', 'learned'),
        help="Type of positional embedding to use on top of the image features"
    )

    # * Transformer
    parser.add_argument('--enc_layers',
                        default=6,
                        type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers',
                        default=6,
                        type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument(
        '--dim_feedforward',
        default=2048,
        type=int,
        help=
        "Intermediate size of the feedforward layers in the transformer blocks"
    )
    parser.add_argument(
        '--hidden_dim',
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout',
                        default=0.1,
                        type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--enc_dropout',
                        default=0,
                        type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument(
        '--nheads',
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries',
                        default=100,
                        type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks',
                        action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument(
        '--no_aux_loss',
        dest='aux_loss',
        action='store_false',
        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class',
                        default=1,
                        type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox',
                        default=5,
                        type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou',
                        default=2,
                        type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--ce_loss_coef', default=1, type=float)
    parser.add_argument('--set_loss_coef', default=1, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument(
        '--eos_coef',
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir',
                        default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)

    # distributed training parameters
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url',
                        default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)
    print('device:', device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    criterion.to(device)

    model_without_ddp = model
    if args.distributed:
        print('distributed')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('number of params:', n_parameters)
    # print([n for n, p in criterion.named_parameters() if p.requires_grad])
    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr":
            args.lr_backbone,
        },
    ]
    if args.dslp_at == 'criterion':
        # if dslp is at model, there is no need to add this part
        param_dicts.append({"params": criterion.parameters()})

    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args, index_offset=0)
    dataset_val = build_dataset(image_set='val',
                                args=args,
                                index_offset=len(dataset_train))

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,
                                                        args.batch_size,
                                                        drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val,
                                 args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume,
                                                            map_location='cpu',
                                                            check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        try:
            criterion.load_state_dict(checkpoint['criterion'])
        except Exception:
            print('cannot load criterion state...')

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device,
                                              args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval,
                                 output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train,
                                      optimizer, device, epoch,
                                      args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir /
                                        f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        'model': model_without_ddp.state_dict(),
                        'criterion': criterion.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

        # test_stats, coco_evaluator = evaluate(
        #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        # )
        # log_stats = {
        #     **{f'train_{k}': v
        #        for k, v in train_stats.items()}, 'epoch': epoch,
        #     'n_parameters': n_parameters
        # }

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device,
                                              args.output_dir)
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              'epoch': epoch,
        #              'n_parameters': n_parameters}

        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()},
            **{f'test_{k}': v
               for k, v in test_stats.items()}, 'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        #     # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path data/ --lr_drop 180 --epochs 240  --batch_size 16 --output_dir stat/fix_bbox_loss > training_log/fix_bbox_loss.log
