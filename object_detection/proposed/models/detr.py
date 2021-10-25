# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized,
                       nested_tensor_from_tensor_list)
from util.tensor import *

from models.bbox_encoder import MLPLNReLU

from .backbone import build_backbone
from .matcher import HungarianMatcherList, build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


@dataclass
class DETRConfig:
    decoder: str = 'v1'


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self,
                 backbone,
                 transformer,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 conf: DETRConfig = None,
                 matcher: HungarianMatcherList = None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        if conf.decoder == 'v1':
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        elif conf.decoder == 'v2':
            self.class_embed = MLPLNReLU(hidden_dim,
                                         3,
                                         last_relu=False,
                                         last_norm=False,
                                         n_out=num_classes + 1)
            self.bbox_embed = MLPLNReLU(hidden_dim,
                                        3,
                                        last_relu=False,
                                        last_norm=False,
                                        n_out=4)
        elif conf.decoder == 'v3':
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.bbox_embed = nn.Linear(hidden_dim, 4)
        else:
            raise NotImplementedError()

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels,
                                    hidden_dim,
                                    kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # if None, no matcher
        self.matcher = matcher

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        # src (bs, c, h, w)
        src, mask = features[-1].decompose()
        assert mask is not None

        # extra {'loss'}
        hs, memory, extra = self.transformer(
            self.input_proj(src),
            mask,
            self.query_embed.weight,
            pos[-1],
            matcher=self.matcher,
            targets=targets,
            backbone_features=src,
        )

        out = {'pred_features': hs[-1], 'bb': src}
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # print(outputs_class.shape, outputs_coord.shape)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'pred_features': hs[-1],
            'bb': src,
            **extra,
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class,
                                                    outputs_coord, hs, src)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, hs, src):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_features': c, 'bb': src} for c in hs[:-1]]

@dataclass
class SetCriterionConfig:
    # 'all', 'top'
    match_mode: str = 'all'
    use_aux: bool = True
    # auxiliary loss assuming the top matching
    aux_top_matching: bool = False
    aux_top_matching_coef: float = 0.5
    majority_scaling: bool = False
    # for switch tracking
    track_switches: bool = True
    data_size: int = 10_000
    set_size: int = 100


class SwitchTracker(nn.Module):
    def __init__(self, n_row, n_col):
        super().__init__()
        # switch count
        self.n_col = n_col
        self.register_buffer('data', torch.full((n_row, n_col), -1).long())
        self.register_buffer('data_cls', torch.full((n_row, n_col), -1).long())

    def count_switches(self, index, ordering, true_object_mask, classes):
        """
        Args:
            index: (bs, ) data indicies within a batch
            lengths: (bs, )
            ordering: (n, )
            classes: (n, ) classes for each head (inverted from the target using pi)
        """
        # (bs, col)
        new = chunk_by_lengths(ordering.clone(), [self.n_col] * len(index))
        cls = chunk_by_lengths(classes, [self.n_col] * len(index))
        # (bs, col)
        m = chunk_by_lengths(true_object_mask, [self.n_col] * len(index))
        # de-offsetting the ordering
        offset = 0
        for i in range(len(index)):
            new[i] -= offset
            offset += self.n_col
        # count the changes
        tot_changes = 0
        tot_cls_chg = 0
        for i in range(len(index)):
            # data[ idx, : ] != new ==> switches
            changes = (self.data[index[i], m[i]] != new[i][m[i]]).sum().item()
            cls_chg = (self.data_cls[index[i]] != cls[i]).sum().item()
            # update the ordering in data
            self.data[index[i], m[i]] = new[i][m[i]]
            self.data_cls[index[i]] = cls[i]
            tot_changes += changes
            tot_cls_chg += cls_chg
        rate = tot_changes / true_object_mask.sum().item()
        rate_cls = tot_cls_chg / len(classes)
        return rate, rate_cls


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher: HungarianMatcherList, weight_dict,
                 eos_coef, losses, conf: SetCriterionConfig):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        print('match mode:', conf.match_mode)
        self.conf = conf
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.set_size = conf.set_size
        if conf.track_switches:
            # track switches for all layers
            self.switch_trackers = nn.ModuleList([
                SwitchTracker(conf.data_size, self.set_size) for i in range(6)
            ])

    def loss_labels(self,
                    outputs,
                    targets,
                    match_out,
                    weights,
                    log=True,
                    **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert 'pred_logits' in outputs
        src_logits = match_out['pred_class']
        dev = src_logits.device
        target_classes_o = [t["labels"] for t in targets]

        target_ids_tensor = torch.full((int(src_logits.shape[0] // self.set_size), self.set_size),
                                       self.num_classes,
                                       dtype=torch.long).to(dev)
        for idx, classes in enumerate(target_classes_o):
            target_ids_tensor[idx, :len(classes)] = classes
        target_ids_tensor = target_ids_tensor.flatten()

        weight_class = torch.ones(len(src_logits)).to(dev)
        weight_class[target_ids_tensor == self.num_classes] *= self.eos_coef
        loss_ce = F.cross_entropy(src_logits,
                                  target_ids_tensor,
                                  reduction='none')
        if weights is None:
            loss_ce = (loss_ce * weight_class).sum() / weight_class.sum()
        else:
            loss_ce = (loss_ce * weight_class *
                       weights).sum() / (weight_class * weights).sum()

        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits,
                                                   target_ids_tensor)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, match_out, num_boxes,
                         **kwargs):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = match_out['pred_class']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets],
                                      device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum()
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, match_out, num_boxes,
                   true_object_loc, weights):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        # assert 'pred_boxes' in outputs
        src_boxes = match_out['pred_box']
        target_boxes = torch.cat([t["boxes"] for t in targets], dim=0)

        loss_bbox = F.l1_loss(src_boxes[true_object_loc],
                              target_boxes,
                              reduction='none')

        if weights is None:
            loss_bbox = loss_bbox.sum() / num_boxes
        else:
            loss_bbox = (loss_bbox * weights[true_object_loc].unsqueeze(-1)
                         ).sum() / weights[true_object_loc].sum()

        losses = {}
        losses['loss_bbox'] = loss_bbox

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes[true_object_loc]),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))

        if weights is None:
            loss_giou = loss_giou.sum() / num_boxes
        else:
            loss_giou = (loss_giou * weights[true_object_loc]
                         ).sum() / weights[true_object_loc].sum()

        losses['loss_giou'] = loss_giou
        return losses

    def loss_masks(self, outputs, targets, match_out, num_boxes, **kwargs):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        raise NotImplementedError()

        src_idx = self._get_src_permutation_idx(match_out)
        tgt_idx = self._get_tgt_permutation_idx(match_out)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None],
                                size=target_masks.shape[-2:],
                                mode="bilinear",
                                align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks,
                                            num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self,
                 loss,
                 outputs,
                 targets,
                 match_out,
                 num_boxes,
                 true_object_loc,
                 weights=None,
                 **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs=outputs,
                              targets=targets,
                              match_out=match_out,
                              num_boxes=num_boxes,
                              true_object_loc=true_object_loc,
                              weights=weights,
                              **kwargs)

    def forward(self, outputs, targets, index, class_embed: nn.Module,
                bbox_embed: nn.Module):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'
        }

        if self.matcher is None:
            losses = outputs['loss']
            PIs = outputs['pi']

        # calculate the true object loc
        tgt_ids = [v["labels"] for v in targets]
        batch_size = len(targets)
        target_ids_tensor = torch.full((batch_size, self.set_size), self.num_classes, dtype=torch.long)
        for idx, classes in enumerate(tgt_ids):
            target_ids_tensor[idx, :len(classes)] = classes
        target_ids_tensor = target_ids_tensor.flatten()
        true_object_loc = torch.where(target_ids_tensor != self.num_classes)
        true_object_mask = torch.zeros(len(target_ids_tensor)).bool()
        true_object_mask[true_object_loc] = True

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes],
                                    dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        def predict(feat):
            pred_class = class_embed(feat)  # (bs  * set_size, n_class)
            pred_box = bbox_embed(feat).sigmoid()  # (bs *set_size, 4)
            return {'pred_class': pred_class, 'pred_box': pred_box}

        #########################
        Inputs = outputs['aux_outputs'] + [outputs_without_aux]
        Outputs = [None] * len(Inputs)

        # predict and match for all layers
        dists_bias = None
        for i in range(len(Inputs)):
            if self.matcher is not None:
                # DSLP at criterion
                # { 'bb', 'pred_features' }
                out = self.matcher.layers[i].forward(Inputs[i],
                                                     targets,
                                                     dists_bias=dists_bias)
                if self.matcher.conf.cumulative_dists:
                    dists_bias = out['dists']
            else:
                # DSLP at model
                # (bs, k, hid)
                feat = Inputs[i]['pred_features']
                # (n, hid)
                feat = feat.flatten(0, 1)
                out = {
                    'loss': losses[i],
                    'feat': feat[PIs[i]],
                    'pi': PIs[i],
                }
            out.update(predict(out['feat']))
            Outputs[i] = out

        top_pi = Outputs[-1]['pi']
        pred = Outputs[-1]
        dev = Outputs[-1]['pred_box'].device

        PIs = torch.stack([Outputs[i]['pi'] for i in range(len(Outputs))],
                          dim=0).to(dev)
        InvPIs = []
        for pi in PIs:
            inv = torch.zeros_like(pi)
            inv[pi] = torch.arange(len(pi), device=inv.device)
            InvPIs.append(inv)

        # (layers, n)
        conformity = calculate_conformity(PIs)
        # (layers, n)
        head_target_ids = torch.stack(
            [target_ids_tensor[inv] for inv in InvPIs]).to(PIs.device)
        class_conformity = calculate_conformity(head_target_ids)

        switch_counts = {}
        if self.conf.track_switches:
            # update switches
            if index is not None:
                # NOTE: during evaluation the index is None
                for i in range(len(Inputs)):
                    switch, class_switch = self.switch_trackers[
                        i].count_switches(index=index,
                                          ordering=PIs[i],
                                          true_object_mask=true_object_mask,
                                          classes=head_target_ids[i])
                    switch_counts[f'switch_{i}'] = switch
                    switch_counts[f'switch_class_{i}'] = class_switch

        if self.conf.majority_scaling:
            # scale the weights of non majority down
            # (layers, n)
            # (layers, n)
            W = conformity
        else:
            W = [None] * len(Inputs)

        losses = {}
        for i in range(len(Inputs)):
            if i == len(Inputs) - 1:
                for loss in self.losses:
                    losses.update(
                        self.get_loss(loss,
                                      outputs=Inputs[i],
                                      targets=targets,
                                      match_out=Outputs[i],
                                      num_boxes=num_boxes,
                                      true_object_loc=true_object_loc,
                                      weights=W[i]))
                losses["loss_set"] = Outputs[i]['loss']
            else:
                if self.conf.use_aux:
                    for loss in self.losses:
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss,
                                               outputs=Inputs[i],
                                               targets=targets,
                                               match_out=Outputs[i],
                                               num_boxes=num_boxes,
                                               true_object_loc=true_object_loc,
                                               weights=W[i],
                                               **kwargs)
                        l_dict['loss_set'] = Outputs[i]['loss']
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

                    if self.conf.aux_top_matching:
                        # doesn't do the hungarian
                        # used the top's pi instead
                        out = self.matcher.layers[i](Inputs[i],
                                                     targets,
                                                     pi=top_pi)
                        out.update(predict(out['feat']))

                        for loss in self.losses:
                            if loss == 'masks':
                                # Intermediate masks losses are too costly to compute, we ignore them.
                                continue
                            kwargs = {}
                            if loss == 'labels':
                                # Logging is enabled only for the last layer
                                kwargs = {'log': False}
                            l_dict = self.get_loss(
                                loss,
                                outputs=Inputs[i],
                                targets=targets,
                                match_out=out,
                                num_boxes=num_boxes,
                                true_object_loc=true_object_loc,
                                weights=W[i],
                                **kwargs)
                            l_dict['loss_set'] = out['loss']
                            l_dict = {
                                k + f'_{i}': v
                                for k, v in l_dict.items()
                            }
                            # mixin the aux_top losses
                            coef = self.conf.aux_top_matching_coef
                            for k, v in l_dict.items():
                                losses[k] = (1 - coef) * losses[k] + coef * v

        conformity_all = conformity.mean(dim=0)
        conformity_all = conformity_all[true_object_mask].mean()
        class_conform_all = class_conformity.mean()

        # extra stats
        extra = {
            'conformity': conformity_all.item(),
            'class_conformity': class_conform_all.item(),
            **switch_counts,
        }

        return losses, pred, extra


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{
            'scores': s,
            'labels': l,
            'boxes': b
        } for s, l, b in zip(scores, labels, boxes)]

        return results


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


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    num_classes = 11
    args.num_classes = num_classes

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    matcher = build_matcher(args)

    dslp_at = args.dslp_at
    print('dslp_at:', dslp_at)
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        conf=DETRConfig(decoder=args.decoder),
        # supply the matcher when dslp is at model
        matcher=matcher if dslp_at == 'model' else None,
    )
    print('decoder:', args.decoder)
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    weight_dict = {
        'loss_ce': args.ce_loss_coef,
        'loss_bbox': args.bbox_loss_coef
    }
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    print('aux_coef:', args.aux_coef)
    print('aux set coef:', args.aux_set_coef)
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                k + f'_{i}': v * args.aux_coef
                for k, v in weight_dict.items()
            })
        weight_dict.update(aux_weight_dict)

    weight_dict['loss_set'] = args.set_loss_coef
    weight_dict['loss_set_0'] = args.set_loss_coef * args.aux_set_coef
    weight_dict['loss_set_1'] = args.set_loss_coef * args.aux_set_coef
    weight_dict['loss_set_2'] = args.set_loss_coef * args.aux_set_coef
    weight_dict['loss_set_3'] = args.set_loss_coef * args.aux_set_coef
    weight_dict['loss_set_4'] = args.set_loss_coef * args.aux_set_coef

    print('set loss coef:', args.set_loss_coef)

    print('loss weights:', weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(
        num_classes,
        # not supply matcher if dslp is not at criterion
        matcher=matcher if dslp_at == 'criterion' else None,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        conf=SetCriterionConfig(match_mode=args.match_mode,
                                use_aux=bool(args.use_aux),
                                aux_top_matching=bool(args.aux_top_matching),
                                aux_top_matching_coef=args.aux_top_coef,
                                majority_scaling=bool(args.majority_scaling),
                                track_switches=bool(args.track_switches)),
    )
    print('track switch:', criterion.conf.track_switches)
    print('use_aux:', criterion.conf.use_aux)
    print('aux_top:', criterion.conf.aux_top_matching)
    print('aux_top_coef:', criterion.conf.aux_top_matching_coef)
    print('majority scaling:', criterion.conf.majority_scaling)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map,
                                                             threshold=0.85)

    return model, criterion, postprocessors


def calculate_conformity(PIs):
    """
    how much the PI of each layer conform with other layers.

    Args:
        PIs: (layers, n)
    """
    conform = []
    for i in range(len(PIs)):
        # (n, )
        avg = (PIs[i] == PIs).float().mean(dim=0)
        conform.append(avg)
    # (layers, n)
    conform = torch.stack(conform)
    return conform
