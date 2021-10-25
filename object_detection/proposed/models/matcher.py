# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from .hungarian import *
from .bbox_encoder import *


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@dataclass
class HungarianMatcherConfig:
    num_classes : int
    mode: str
    dropout: float
    safe_coef: float
    w_pred_gt: float
    w_gt_pred: float
    ignore_nocls_bbox: bool
    distance_both_ways: bool
    aux_mode: str = None
    aux_safe_coef: float = None
    n_hid: int = 256
    encoder: str = 'v1'
    enc_use_context: bool = True
    enc_context_type: str = None
    enc_mlp_pre_reduce: bool = True
    enc_mlp_post_reduce: bool = True
    enc_common_layer: int = 1
    # concat input mode
    use_concat: bool = False
    n_repeat: int = None
    cumulative_dists: bool = False
    sum_over_n: bool = False

    def make_matcher(self):
        return HungarianMatcher(self)


class HungarianMatcherList(nn.Module):
    """multi-layer hungarian matcher. All layers share the prediction head."""
    def __init__(self, n_layer: int, conf: HungarianMatcherConfig, share: str):
        super().__init__()
        self.conf = conf

        confs = [deepcopy(conf) for i in range(n_layer)]

        # change the safe_coef of lower layers
        if self.conf.aux_safe_coef is not None:
            for i in range(n_layer - 1):
                confs[i].safe_coef = self.conf.aux_safe_coef
        if self.conf.aux_mode is not None:
            for i in range(n_layer - 1):
                confs[i].mode = self.conf.aux_mode

        if share == 'none':
            self.layers = nn.ModuleList(
                [confs[i].make_matcher() for i in range(n_layer)])
        elif share == 'emb':
            self.layers = nn.ModuleList(
                [confs[i].make_matcher() for i in range(n_layer)])
            # share emb, layers
            for i in range(1, n_layer):
                self.layers[i].bbox_encoder.cls_emb = self.layers[
                    0].bbox_encoder.cls_emb
                self.layers[i].bbox_encoder.xy_emb = self.layers[
                    0].bbox_encoder.xy_emb
                self.layers[i].bbox_encoder.layers = self.layers[
                    0].bbox_encoder.layers
        elif share == 'emb+ctx':
            self.layers = nn.ModuleList(
                [confs[i].make_matcher() for i in range(n_layer)])
            # share emb, layers, ctx
            for i in range(1, n_layer):
                self.layers[i].bbox_encoder.cls_emb = self.layers[
                    0].bbox_encoder.cls_emb
                self.layers[i].bbox_encoder.xy_emb = self.layers[
                    0].bbox_encoder.xy_emb
                self.layers[i].bbox_encoder.layers = self.layers[
                    0].bbox_encoder.layers
                # ctx
                self.layers[i].bbox_encoder.ctx_layers = self.layers[
                    0].bbox_encoder.ctx_layers
                self.layers[i].bbox_encoder.post_ctx_layers = self.layers[
                    0].bbox_encoder.post_ctx_layers
        elif share == 'emb+out':
            self.layers = nn.ModuleList(
                [confs[i].make_matcher() for i in range(n_layer)])
            # share emb, layers, ctx
            for i in range(1, n_layer):
                self.layers[i].bbox_encoder.cls_emb = self.layers[
                    0].bbox_encoder.cls_emb
                self.layers[i].bbox_encoder.xy_emb = self.layers[
                    0].bbox_encoder.xy_emb
                self.layers[i].bbox_encoder.layers = self.layers[
                    0].bbox_encoder.layers
                self.layers[i].bbox_encoder.output = self.layers[
                    0].bbox_encoder.output
        else:
            raise NotImplementedError()

    def forward(self, outputs, targets):
        return self.layers[-1].forward(outputs, targets)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, conf: HungarianMatcherConfig):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.conf = conf
        self.mode = conf.mode
        self.num_classes = conf.num_classes
        self.set_size = 100
        if conf.encoder == 'v1':
            assert not conf.use_concat, "the legacy encoder doesn't support concat mode"
            self.cls_emb = nn.Sequential(
                nn.Embedding(self.num_classes + 1, conf.n_hid),
                MLPLNReLU(conf.n_hid, 3, last_norm=True, last_relu=False))
            self.bbox_emb = MLPLNReLU(conf.n_hid,
                                      3,
                                      last_norm=True,
                                      last_relu=False,
                                      n_inp=4)
            self.ctx_emb = MLPLNReLU(conf.n_hid,
                                     3,
                                     last_norm=True,
                                     last_relu=False,
                                     n_inp=2048)

            self.merge = nn.Linear(256 * 3, 256)
            self.bn_merge = nn.LayerNorm(256)
        else:
            if conf.encoder == 'v2':
                enc_conf = BoxEncoderV2Config(
                    n_cls=self.num_classes + 1,
                    n_hid=conf.n_hid,
                    n_layer=3,
                    n_common_layer=conf.enc_common_layer,
                    use_context=conf.enc_use_context,
                    context_type=conf.enc_context_type,
                    detach_context=conf.enc_context_type != 'context',
                    mlp_pre_reduce=conf.enc_mlp_pre_reduce,
                    mlp_post_reduce=conf.enc_mlp_post_reduce,
                    dropout=conf.dropout,
                    ignore_cls_id=self.num_classes,
                    do_ignore=True,
                )
            elif conf.encoder == 'v3':
                enc_conf = BoxEncoderV3Config(
                    n_cls=self.num_classes + 1,
                    n_hid=conf.n_hid,
                    n_layer=3,
                    n_common_layer=conf.enc_common_layer,
                    use_context=conf.enc_use_context,
                    context_type=conf.enc_context_type,
                    detach_context=conf.enc_context_type != 'context',
                    mlp_pre_reduce=conf.enc_mlp_pre_reduce,
                    mlp_post_reduce=conf.enc_mlp_post_reduce,
                    dropout=conf.dropout,
                )
            else:
                raise NotImplementedError()

            if conf.use_concat:
                self.bbox_encoder = BoxEncoderRepeatConfig(
                    n_repeat=conf.n_repeat, enc_config=enc_conf).make_model()
            else:
                self.bbox_encoder = enc_conf.make_model()

        if conf.mode == 'none':
            self.hungarian_mapper = MSEGCRLatentLossConfig(
                w_loss_br=conf.w_gt_pred,
                w_loss_rb=conf.w_pred_gt,
                gcr_mode='none',
                sum_over_n=self.conf.sum_over_n,
            ).make_loss()
        elif conf.mode == 'gc':
            self.hungarian_mapper = MSEGCRLatentLossConfig(
                w_loss_br=conf.w_gt_pred,
                w_loss_rb=conf.w_pred_gt,
                gcr_mode='gc',
                sum_over_n=self.conf.sum_over_n,
            ).make_loss()
        elif conf.mode == 'gcr':
            self.hungarian_mapper = MSEGCRLatentLossConfig(
                w_loss_br=conf.w_gt_pred,
                w_loss_rb=conf.w_pred_gt,
                gcr_mode='gcr',
                distance_both_ways=conf.distance_both_ways,
                safe_coef=conf.safe_coef,
                sum_over_n=self.conf.sum_over_n,
            ).make_loss()
        else:
            raise NotImplementedError()

        # the prediction head is supplied at runtime

    # @torch.no_grad()
    def forward(self, outputs, targets, pi=None, dists_bias=None):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

            pi: if present, enforce the matching no need for Hungarian
            dists_bias: if present, will be added to the calculated dists

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        # Also concat the target labels and boxes

        ###############set up GT format##############################

        img_features = None
        memory_features = None
        if 'bb' in outputs:
            # (bs, c, h, w)
            img_features = outputs["bb"]
            m = torch.nn.AdaptiveAvgPool2d((1, 1))
            # (bs, c)
            img_features = m(img_features)[:, :, 0, 0]
        elif 'memory' in outputs:
            # (bs, k, hid)
            memory = outputs['memory']
            # (bs, hid)
            memory_features = memory.mean(dim=1)
        else:
            raise NotImplementedError()

        # (bs, 100, hid)
        pred_features = outputs['pred_features']
        dev = pred_features.device
        batch_size = pred_features.shape[0]

        # image features
        tgt_ids = [v["labels"] for v in targets]
        tgt_bbox = [v["boxes"] for v in targets]
        # test_bbox = torch.cat([v["boxes"] for v in targets])

        target_ids_tensor = torch.full((batch_size, self.set_size), self.num_classes, dtype=torch.long)
        target_bbox_tensor = torch.full((batch_size, self.set_size, 4),
                                        0,
                                        dtype=torch.float32)

        for idx, (classes, bbox) in enumerate(zip(tgt_ids, tgt_bbox)):
            target_ids_tensor[idx, :len(classes)] = classes
            target_bbox_tensor[idx, :len(bbox)] = bbox

        target_ids_tensor = target_ids_tensor.flatten()
        target_bbox_tensor = target_bbox_tensor.flatten(0, 1)
        true_object_loc = torch.where(target_ids_tensor != self.num_classes)
        no_object_loc = torch.where(target_ids_tensor == self.num_classes)

        ############################################################
        sizes = torch.LongTensor([self.set_size] * batch_size).to(dev)
        # (bs * 100, hid)
        features = pred_features.flatten(0, 1)
        pred_size = torch.LongTensor([self.set_size] * batch_size).to(dev)

        if self.conf.encoder == 'v1':
            source_feature_tensor = torch.zeros((batch_size, self.set_size, 2048),
                                                dtype=torch.float32)
            for idx, f in enumerate(img_features):
                source_feature_tensor[idx] = f
            source_feature_tensor = source_feature_tensor.flatten(0, 1).to(dev)

            x1 = self.cls_emb(target_ids_tensor.to(dev))
            x2 = self.bbox_emb(target_bbox_tensor.to(dev))
            if self.conf.ignore_nocls_bbox:
                # should be better?
                x2[no_object_loc] = 0
            x3 = self.ctx_emb(source_feature_tensor.to(dev))
            gt = torch.cat((x1, x2, x3), 1)
            gt = self.bn_merge(self.merge(gt))
        elif self.conf.encoder in ['v2', 'v3']:
            # (n, n_hid)
            gt = self.bbox_encoder.forward(
                xywh=target_bbox_tensor.to(dev),
                cls_id=target_ids_tensor.to(dev),
                lengths=sizes,
                context=img_features,
                memory=memory_features,
                context_R=features,
                context_R_lengths=pred_size,
            )
        else:
            raise NotImplementedError()

        # latent loss
        out = self.hungarian_mapper.forward(gt,
                                            sizes,
                                            features,
                                            pred_size,
                                            cols=pi,
                                            dists_bias=dists_bias)
        pred = out.R_pi

        # outputs_class = class_embed(pred)  # (bs  * 100, n_class)
        # outputs_coord = bbox_embed(pred).sigmoid()  # (bs *100, 4)
        # outputs_coord = outputs_coord[true_object_loc]  # (sum_k, 4)

        return {
            # 'pred_class': outputs_class,
            # 'pred_box': outputs_coord,
            'true_object_loc': true_object_loc,
            'feat': pred,
            'pi': out.R_i,
            'loss': out.loss,
            'dists': out.dists,
        }

def build_matcher(args):

    conf = HungarianMatcherConfig(
        num_classes = args.num_classes,
        encoder=args.enc,
        mode=args.mode,
        dropout=args.enc_dropout,
        safe_coef=args.safe_coef,
        aux_mode=args.aux_mode,
        aux_safe_coef=args.aux_safe_coef,
        w_pred_gt=args.w_pred_gt,
        w_gt_pred=args.w_gt_pred,
        ignore_nocls_bbox=bool(args.ignore_nocls_bbox),
        distance_both_ways=bool(args.distance_both_ways),
        enc_use_context=bool(args.enc_context),
        enc_context_type=args.enc_context_type,
        enc_mlp_pre_reduce=bool(args.enc_mlp_pre_reduce),
        enc_mlp_post_reduce=bool(args.enc_mlp_post_reduce),
        enc_common_layer=args.enc_common_layer,
        use_concat=args.match_mode == 'concat',
        n_repeat=6,
        cumulative_dists=bool(args.cumulative_dists),
        sum_over_n=bool(args.sum_over_n),
    )
    print('enc share:', args.enc_share)
    print('enc:', conf.encoder)
    print('mode:', conf.mode, 'aux:', conf.aux_mode)
    print('safe coef:', conf.safe_coef, 'aux:', conf.aux_safe_coef)
    print('enc_context', conf.enc_use_context)
    print('enc_context_type:', conf.enc_context_type)
    print('enc_mlp_pre_reduce:', conf.enc_mlp_pre_reduce)
    print('enc_mlp_post_reduce:', conf.enc_mlp_post_reduce)
    print('enc_common_layer:', conf.enc_common_layer)
    print('safe_coef:', conf.safe_coef)
    print('w:', (conf.w_pred_gt, conf.w_gt_pred))
    print('ignore nocls bbox:', conf.ignore_nocls_bbox)
    print('distance both ways:', conf.distance_both_ways)
    print('use concat:', conf.use_concat)
    print('enc_dropout:', conf.dropout)
    print('cumulative dists:', conf.cumulative_dists)
    print('sum_over_n:', conf.sum_over_n)

    if conf.use_concat:
        return HungarianMatcher(conf)
    else:
        return HungarianMatcherList(
            n_layer=6,
            conf=conf,
            share=args.enc_share,
        )
