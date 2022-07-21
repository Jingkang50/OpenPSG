# ---------------------------------------------------------------
# relation_util.py
# Set-up time: 2020/5/7 下午11:13
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import copy
from collections import defaultdict

# import anytree
import numpy as np
import torch
import torch.nn as nn
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from torch.nn import functional as F


class Result(object):
    """ little container class for holding the detection result
        od: object detector, rm: rel model"""
    def __init__(
        self,
        bboxes=None,  # gt bboxes / OD: det bboxes
        dists=None,  # OD: predicted dists
        labels=None,  # gt labels / OD: det labels
        masks=None,  # gt masks  / OD: predicted masks
        formatted_masks=None,  # OD: Transform the masks for object detection evaluation
        points=None,  # gt points / OD: predicted points
        rels=None,  # gt rel triplets / OD: sampled triplets (training) with target rel labels
        key_rels=None,  # gt key rels
        relmaps=None,  # gt relmaps
        refine_bboxes=None,  # RM: refined object bboxes (score is changed)
        formatted_bboxes=None,  # OD: Transform the refine_bboxes for object detection evaluation
        refine_scores=None,  # RM: refined object scores (before softmax)
        refine_dists=None,  # RM: refined object dists (after softmax)
        refine_labels=None,  # RM: refined object labels
        target_labels=None,  # RM: assigned object labels for training the relation module.
        rel_scores=None,  # RM: predicted relation scores (before softmax)
        rel_dists=None,  # RM: predicted relation prob (after softmax)
        triplet_scores=None,  # RM: predicted triplet scores (the multiplication of sub-obj-rel scores)
        ranking_scores=None,  # RM: predicted ranking scores for rank the triplet
        rel_pair_idxes=None,  # gt rel_pair_idxes / RM: training/testing sampled rel_pair_idxes
        rel_labels=None,  # gt rel_labels / RM: predicted rel labels
        target_rel_labels=None,  # RM: assigned target rel labels
        target_key_rel_labels=None,  # RM: assigned target key rel labels
        saliency_maps=None,  # SAL: predicted or gt saliency map
        attrs=None,  # gt attr
        rel_cap_inputs=None,  # gt relational caption inputs
        rel_cap_targets=None,  # gt relational caption targets
        rel_ipts=None,  # gt relational importance scores
        tgt_rel_cap_inputs=None,  # RM: assigned target relational caption inputs
        tgt_rel_cap_targets=None,  # RM: assigned target relational caption targets
        tgt_rel_ipts=None,  # RM: assigned target relational importance scores
        rel_cap_scores=None,  # RM: predicted relational caption scores
        rel_cap_seqs=None,  # RM: predicted relational seqs
        rel_cap_sents=None,  # RM: predicted relational decoded captions
        rel_ipt_scores=None,  # RM: predicted relational caption ipt scores
        cap_inputs=None,
        cap_targets=None,
        cap_scores=None,
        cap_scores_from_triplet=None,
        alphas=None,
        rel_distribution=None,
        obj_distribution=None,
        word_obj_distribution=None,
        cap_seqs=None,
        cap_sents=None,
        img_shape=None,
        scenes=None,  # gt scene labels
        target_scenes=None,  # target_scene labels
        add_losses=None,  # For Recording the loss except for final object loss and rel loss, e.g.,
        # use in causal head or VCTree, for recording auxiliary loss
        head_spec_losses=None,  # For method-specific loss
        pan_results=None,
    ):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def is_none(self):
        return all(
            [v is None for k, v in self.__dict__.items() if k != 'self'])

    # HACK: To turn this object into an iterable
    def __len__(self):
        return 1

    # HACK:
    def __getitem__(self, i):
        return self

    # HACK:
    def __iter__(self):
        yield self


class PostProcessor(nn.Module):
    """Obtain the final relation information for evaluation."""
    def __init__(self):
        """
        Arguments:

        """
        super(PostProcessor, self).__init__()

    def forward(self, det_result, key_first=False):
        """
        Arguments:
            det_result

        Returns:
            det_result: add the
        """
        if det_result.refine_scores is None:
            return det_result
        relation_logits, finetune_obj_logits = det_result.rel_scores, det_result.refine_scores
        rel_pair_idxes = det_result.rel_pair_idxes
        ranking_scores = det_result.ranking_scores

        finetune_labels, finetune_dists, finetune_bboxes, \
        rels, rel_dists, prop_rel_pair_idxes, prop_rel_labels, prop_rel_scores, triplet_scores = \
            [], [], [], [], [], [], [], [], []
        prop_ranking_scores = None if ranking_scores is None else []

        for i, (rel_logit, obj_logit, rel_pair_idx, bbox) in enumerate(
                zip(relation_logits, finetune_obj_logits, rel_pair_idxes,
                    det_result.bboxes)):
            obj_class_prob = F.softmax(obj_logit, -1)
            obj_class_prob[:, 0] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]

            obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
            obj_pred = obj_pred + 1

            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            finetune_labels.append(obj_class)
            finetune_dists.append(obj_class_prob)
            if bbox.shape[1] == 4:
                bbox = torch.cat((bbox, obj_scores[:, None]), dim=-1)
            else:
                bbox[:, -1] = obj_scores
            finetune_bboxes.append(bbox)

            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            rel_class_prob = F.softmax(rel_logit, -1)
            rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
            rel_class = rel_class + 1
            # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
            triple_scores = rel_scores * obj_scores0 * obj_scores1
            if key_first and ranking_scores is not None:
                triple_scores *= ranking_scores[i]
            _, sorting_idx = torch.sort(triple_scores.view(-1),
                                        dim=0,
                                        descending=True)
            triple_scores = triple_scores.view(-1)[sorting_idx].contiguous()
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]
            rel_logit = rel_logit[sorting_idx]
            if key_first and ranking_scores is not None:
                prop_ranking_scores.append(ranking_scores[i][sorting_idx])

            prop_rel_pair_idxes.append(rel_pair_idx)
            prop_rel_labels.append(rel_labels)
            prop_rel_scores.append(rel_logit)
            rel = torch.cat((rel_pair_idx, rel_labels[:, None]), dim=-1)
            rels.append(rel)
            rel_dists.append(rel_class_prob)
            triplet_scores.append(triple_scores)

        det_result.refine_bboxes = finetune_bboxes
        det_result.refine_dists = finetune_dists
        det_result.refine_labels = finetune_labels
        det_result.rels = rels
        det_result.rel_dists = rel_dists
        det_result.rel_pair_idxes = prop_rel_pair_idxes
        det_result.triplet_scores = triplet_scores
        det_result.rel_labels = prop_rel_labels
        det_result.rel_scores = prop_rel_scores
        det_result.ranking_scores = prop_ranking_scores
        return det_result


class DemoPostProcessor(object):
    """This API is used for obtaining the final information for demonstrating
    the scene graphs.

    It's usually invoked after the PostProcessor. Especially applying NMS to
    suppress the repetition.
    """
    def __init__(self):
        super(DemoPostProcessor, self).__init__()

    def filter_AB_rels(self, det_result):
        new_rel_pair_idxes = []
        rel_pair_idxes = det_result.rel_pair_idxes
        keep_rel_idxes = []
        for idx, pair in enumerate(rel_pair_idxes):
            subj, obj = pair[0], pair[1]
            pair = pair.tolist()
            if pair in new_rel_pair_idxes or [obj, subj] in new_rel_pair_idxes:
                continue
            new_rel_pair_idxes.append(pair)
            keep_rel_idxes.append(idx)
        new_rel_pair_idxes = np.array(new_rel_pair_idxes).astype(np.int32)
        det_result.rel_pair_idxes = new_rel_pair_idxes
        det_result.rel_labels = det_result.rel_labels[keep_rel_idxes]
        if len(keep_rel_idxes) > 0:
            det_result.rels = np.hstack(
                (det_result.rel_pair_idxes, det_result.rel_labels[:, None]))
        else:
            det_result.rels = np.array([]).astype(np.int32)
        det_result.rel_dists = det_result.rel_dists[keep_rel_idxes]
        det_result.triplet_scores = det_result.triplet_scores[keep_rel_idxes]
        return det_result

    def filter_rels_by_duplicated_names(self, det_result):
        new_rel_pair_idxes = []
        rel_pair_idxes = det_result.rel_pair_idxes
        refine_labels = det_result.refine_labels
        keep_rel_idxes = []
        for idx, pair in enumerate(rel_pair_idxes):
            subj, obj = pair[0], pair[1]
            if refine_labels[subj] == refine_labels[obj]:
                continue
            new_rel_pair_idxes.append(pair)
            keep_rel_idxes.append(idx)
        new_rel_pair_idxes = np.array(new_rel_pair_idxes).astype(np.int32)
        det_result.rel_pair_idxes = new_rel_pair_idxes
        det_result.rel_labels = det_result.rel_labels[keep_rel_idxes]
        if len(keep_rel_idxes) > 0:
            det_result.rels = np.hstack(
                (det_result.rel_pair_idxes, det_result.rel_labels[:, None]))
        else:
            det_result.rels = np.array([]).astype(np.int32)
        det_result.rel_dists = det_result.rel_dists[keep_rel_idxes]
        det_result.triplet_scores = det_result.triplet_scores[keep_rel_idxes]
        return det_result

    def filter_nonoverlap_rels(self, det_result, must_overlap_predicates=None):
        refine_bboxes = det_result.refine_bboxes
        refine_labels = det_result.refine_labels
        ious = bbox_overlaps(refine_bboxes[:, :-1], refine_bboxes[:, :-1])
        # refine_logits = det_result.refine_scores  # N * (C+1)
        new_rel_pair_idxes = []
        rel_pair_idxes = det_result.rel_pair_idxes
        rel_labels = det_result.rel_labels
        rel_dists = det_result.rel_dists
        keep_rel_idxes = []
        for idx, (pair, predicate) in enumerate(zip(rel_pair_idxes,
                                                    rel_labels)):
            subj, obj = pair[0], pair[1]
            iou = ious[subj, obj]
            if must_overlap_predicates is not None and predicate in must_overlap_predicates and iou <= 0:
                continue
            new_rel_pair_idxes.append(pair)
            keep_rel_idxes.append(idx)
        new_rel_pair_idxes = np.array(new_rel_pair_idxes).astype(np.int32)
        det_result.rel_pair_idxes = new_rel_pair_idxes
        det_result.rel_labels = det_result.rel_labels[keep_rel_idxes]
        if len(keep_rel_idxes) > 0:
            det_result.rels = np.hstack(
                (det_result.rel_pair_idxes, det_result.rel_labels[:, None]))
        else:
            det_result.rels = np.array([]).astype(np.int32)
        det_result.rel_dists = det_result.rel_dists[keep_rel_idxes]
        det_result.triplet_scores = det_result.triplet_scores[keep_rel_idxes]
        return det_result

    def filter_duplicate_triplets(self, det_result, vocab_objects,
                                  vocab_predicates):
        all_triplets = []
        new_rel_pair_idxes = []
        refine_labels = det_result.refine_labels
        rel_pair_idxes = det_result.rel_pair_idxes
        rel_labels = det_result.rel_labels
        rel_dists = det_result.rel_dists
        keep_rel_idxes = []
        for idx, (pair, predicate) in enumerate(zip(rel_pair_idxes,
                                                    rel_labels)):
            triplet = [
                vocab_objects[refine_labels[pair[0]]],
                vocab_predicates[predicate],
                vocab_objects[refine_labels[pair[1]]]
            ]
            if triplet in all_triplets:
                continue
            new_rel_pair_idxes.append(pair)
            keep_rel_idxes.append(idx)
            all_triplets.append(triplet)
        new_rel_pair_idxes = np.array(new_rel_pair_idxes).astype(np.int32)
        det_result.rel_pair_idxes = new_rel_pair_idxes
        det_result.rel_labels = det_result.rel_labels[keep_rel_idxes]
        if len(keep_rel_idxes) > 0:
            det_result.rels = np.hstack(
                (det_result.rel_pair_idxes, det_result.rel_labels[:, None]))
        else:
            det_result.rels = np.array([]).astype(np.int32)
        det_result.rel_dists = det_result.rel_dists[keep_rel_idxes]
        det_result.triplet_scores = det_result.triplet_scores[keep_rel_idxes]
        return det_result

    def filter_rels_by_num(self, det_result, num):
        det_result.rel_pair_idxes = det_result.rel_pair_idxes[:num]
        det_result.rel_labels = det_result.rel_labels[:num]
        if len(det_result.rel_labels) > 0:
            det_result.rels = np.hstack(
                (det_result.rel_pair_idxes, det_result.rel_labels[:, None]))
        else:
            det_result.rels = np.array([]).astype(np.int32)
        det_result.rel_dists = det_result.rel_dists[:num]
        det_result.triplet_scores = det_result.triplet_scores[:num]
        return det_result

    def filtered_rels_by_mincover(self, det_result):
        new_rel_pair_idxes = []
        rel_pair_idxes = det_result.rel_pair_idxes
        rel_labels = det_result.rel_labels
        rel_dists = det_result.rel_dists
        keep_rel_idxes = []
        covered_objects = []
        for idx, (pair, predicate) in enumerate(zip(rel_pair_idxes,
                                                    rel_labels)):
            if pair[0] in covered_objects and pair[1] in covered_objects:
                continue
            if pair[0] not in covered_objects:
                covered_objects.append(pair[0])
            if pair[1] not in covered_objects:
                covered_objects.append(pair[1])
            new_rel_pair_idxes.append(pair)
            keep_rel_idxes.append(idx)
        new_rel_pair_idxes = np.array(new_rel_pair_idxes).astype(np.int32)
        det_result.rel_pair_idxes = new_rel_pair_idxes
        det_result.rel_labels = det_result.rel_labels[keep_rel_idxes]
        if len(keep_rel_idxes) > 0:
            det_result.rels = np.hstack(
                (det_result.rel_pair_idxes, det_result.rel_labels[:, None]))
        else:
            det_result.rels = np.array([]).astype(np.int32)
        det_result.rel_dists = det_result.rel_dists[keep_rel_idxes]
        det_result.triplet_scores = det_result.triplet_scores[keep_rel_idxes]
        return det_result

    def clean_relations_via_objects(self, keep_obj_ids, det_result):
        det_result.refine_labels = det_result.refine_labels[keep_obj_ids]
        det_result.refine_bboxes = det_result.refine_bboxes[keep_obj_ids]
        det_result.refine_dists = det_result.refine_dists[keep_obj_ids]
        old_to_new = dict(
            zip(keep_obj_ids.tolist(), list(range(len(keep_obj_ids)))))
        new_rel_pair_idxes = []
        rel_pair_idxes = det_result.rel_pair_idxes
        keep_rel_idxes = []
        for idx, rel in enumerate(rel_pair_idxes):
            if rel[0] not in keep_obj_ids or rel[1] not in keep_obj_ids:
                continue
            new_rel_pair_idxes.append([old_to_new[rel[0]], old_to_new[rel[1]]])
            keep_rel_idxes.append(idx)
        new_rel_pair_idxes = np.array(new_rel_pair_idxes).astype(np.int32)
        det_result.rel_pair_idxes = new_rel_pair_idxes
        det_result.rel_labels = det_result.rel_labels[keep_rel_idxes]
        if len(keep_rel_idxes) > 0:
            det_result.rels = np.hstack(
                (det_result.rel_pair_idxes, det_result.rel_labels[:, None]))
        else:
            det_result.rels = np.array([]).astype(np.int32)
        det_result.rel_dists = det_result.rel_dists[keep_rel_idxes]
        det_result.triplet_scores = det_result.triplet_scores[keep_rel_idxes]
        return det_result

    def forward(self,
                det_result,
                vocab_objects,
                vocab_predicates,
                object_thres=0.01,
                nms_thres=0.1,
                ignore_classes=None,
                must_overlap_predicates=None,
                max_rel_num=None):
        # TODO: Here we only process the box. Any other things related with box, e.g., masks, points are not processed yet.

        # directly ignore objects:
        keep_obj_ids = np.where(
            np.isin(det_result.refine_labels, ignore_classes) == 0)[0]
        det_result = self.clean_relations_via_objects(keep_obj_ids, det_result)

        if len(keep_obj_ids) == 0:
            return det_result

        # apply NMS
        nms_keep_obj_ids, gathered = multiclass_nms_for_cluster(
            det_result.refine_bboxes[:, :-1],
            det_result.refine_bboxes[:, -1],
            det_result.refine_labels,
            nms_thres=nms_thres)
        det_result = self.clean_relations_via_objects(nms_keep_obj_ids,
                                                      det_result)
        if len(nms_keep_obj_ids) == 0:
            return det_result

        # NOTE: This may be not necessary: Suppress the low-score objects
        score_keep_obj_ids = np.where(
            det_result.refine_bboxes[:, -1] >= object_thres)[0]
        det_result = self.clean_relations_via_objects(score_keep_obj_ids,
                                                      det_result)
        if len(score_keep_obj_ids) == 0:
            return det_result

        # Filter the A-B & B-A pairs, keep the pairs with higher scores
        det_result = self.filter_AB_rels(det_result)

        # Filter the rels whose pairs must be overlapped
        det_result = self.filter_nonoverlap_rels(det_result,
                                                 must_overlap_predicates)

        # Filter the duplicate triplets
        det_result = self.filter_duplicate_triplets(det_result, vocab_objects,
                                                    vocab_predicates)

        # Filter the rels by min cover
        det_result = self.filtered_rels_by_mincover(det_result)

        # Filter the rel pairs with the same subj-obj names
        det_result = self.filter_rels_by_duplicated_names(det_result)

        # Control the number of the relations
        num_obj = det_result.refine_bboxes.shape[0]
        rel_num = max_rel_num if max_rel_num is not None else int(
            num_obj * (num_obj - 1) / 2 - num_obj)
        det_result = self.filter_rels_by_num(det_result, rel_num)
        return det_result


def get_box_info(boxes, need_norm=True, size=None):
    """
    input: [batch_size, (x1,y1,x2,y2)]
    size: [h, w]
    output: [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    """
    wh = boxes[:, 2:4] - boxes[:, :2] + 1.0
    center_box = torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)
    box_info = torch.cat((boxes, center_box), 1)
    if need_norm:
        box_info = box_info / float(max(max(size[0], size[1]), 100))
    return box_info


def get_box_pair_info(box1, box2):
    """
    input:
        box1 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
        box2 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    output:
        32-digits: [box1, box2, unionbox, intersectionbox]
    """
    # union box
    unionbox = box1[:, :4].clone()
    unionbox[:, 0] = torch.min(box1[:, 0], box2[:, 0])
    unionbox[:, 1] = torch.min(box1[:, 1], box2[:, 1])
    unionbox[:, 2] = torch.max(box1[:, 2], box2[:, 2])
    unionbox[:, 3] = torch.max(box1[:, 3], box2[:, 3])
    union_info = get_box_info(unionbox, need_norm=False)

    # intersection box
    intersextion_box = box1[:, :4].clone()
    intersextion_box[:, 0] = torch.max(box1[:, 0], box2[:, 0])
    intersextion_box[:, 1] = torch.max(box1[:, 1], box2[:, 1])
    intersextion_box[:, 2] = torch.min(box1[:, 2], box2[:, 2])
    intersextion_box[:, 3] = torch.min(box1[:, 3], box2[:, 3])
    case1 = torch.nonzero(intersextion_box[:, 2].contiguous().view(
        -1) < intersextion_box[:, 0].contiguous().view(-1)).view(-1)
    case2 = torch.nonzero(intersextion_box[:, 3].contiguous().view(
        -1) < intersextion_box[:, 1].contiguous().view(-1)).view(-1)
    intersextion_info = get_box_info(intersextion_box, need_norm=False)
    if case1.numel() > 0:
        intersextion_info[case1, :] = 0
    if case2.numel() > 0:
        intersextion_info[case2, :] = 0
    return torch.cat((box1, box2, union_info, intersextion_info), 1)


def group_regions(result, prior_pairs, thres=0.9):
    """
    Arguments:
            result: (Result object)
            prior_pairs: (List[list]): candidate pair that may be a group
            obj_classes: (List): including the background
        Returns:
            dict: describing the region governing hierarchy
    """
    # NOTE: Extract the RM refined ones.
    bboxes, obj_labels = result.refine_bboxes, result.refine_labels
    region_groups = []
    for boxes, labels in zip(bboxes, obj_labels):
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = boxes.copy()
        num_obj = len(boxes_np)
        if num_obj == 0:
            region_groups.append(None)
            continue
        box_areas = (boxes_np[:, 2] - boxes_np[:, 0] +
                     1) * (boxes_np[:, 3] - boxes_np[:, 1] + 1)
        intersect = bbox_overlaps(boxes, boxes, mode='iof')

        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels.copy()

        region_group = defaultdict(list)
        for i in range(num_obj):
            for j in range(i):
                subj_cls, obj_cls = labels_np[i], labels_np[j]
                subj_area, obj_area = box_areas[i], box_areas[j]
                if [subj_cls, obj_cls] in prior_pairs:
                    # this pair maybe the group ones, check the position
                    if subj_area > obj_area:
                        if intersect[j, i] > thres:
                            if j in region_group:
                                region_group[i] = list(
                                    set(region_group[i] + region_group[j] +
                                        [j]))
                            else:
                                region_group[i].append(j)
                    else:
                        if intersect[i, j] > thres:
                            if i in region_group:
                                region_group[j] = list(
                                    set(region_group[j] + region_group[i] +
                                        [i]))
                                region_group[j].append(i)
        region_groups.append(dict(region_group))
    return region_groups


def get_internal_labels(leaf_labels, hierarchy, vocab):
    leaf_labels_np = leaf_labels.cpu().numpy()
    internal_labels = [[] for _ in leaf_labels_np]
    for idx, leaf_label in enumerate(leaf_labels_np):
        leaf_name = vocab[leaf_label]
        start_node = anytree.search.find(hierarchy,
                                         lambda node: node.id == leaf_name)
        iter_node = start_node
        while iter_node.parent is not None:
            iter_node = iter_node.parent
            internal_labels[idx].append(vocab.index(iter_node.id))
        internal_labels[idx] = torch.from_numpy(
            internal_labels[idx]).to(leaf_labels)
    return internal_labels


def get_pattern_labels(leaf_labels, hierarchy, vocab):
    pattern_labels = []
    for idx, leaf_label in enumerate(leaf_labels):
        leaf_name = vocab[leaf_label]
        start_node = anytree.search.find(hierarchy,
                                         lambda node: node.id == leaf_name)
        iter_node = start_node
        while iter_node.parent.id != 'Root':
            iter_node = iter_node.parent
        pattern_labels.append(vocab.index(iter_node.id))

    pattern_labels = np.array(pattern_labels, dtype=np.int32)
    return pattern_labels


def _topdown_hook(root, output_vector, input_vector, vocab, reduce='avg'):
    if len(root.children) == 0:
        if root.id == 'Root':
            return output_vector
        else:
            output_vector[vocab.index(root.id)] = input_vector[vocab.index(
                root.id)]
            return output_vector
    else:
        gather_values = []
        for c in root.children:
            output_vector = _topdown_hook(c, output_vector, input_vector,
                                          vocab, reduce)
            gather_values.append(output_vector[vocab.index(c.id)][None])
        if reduce == 'avg':
            op = torch.mean
        elif reduce == 'sum':
            op = torch.sum
        elif reduce == 'max':
            op = torch.max
        elif reduce == 'min':
            op = torch.min
        else:
            raise NotImplementedError
        if root.id == 'Root':
            return output_vector
        else:
            output_vector[vocab.index(root.id)] = op(torch.cat(gather_values))
            return output_vector


def top_down_induce(x, hierarchy, vocab, reduce='avg', solveroot=None):
    """The first n elements of vector belong the the first n elements of vocab.

    trick: the input vector name must be "x"!!!!
    """
    vocab_vec = torch.zeros((x.shape[0], len(vocab))).to(x)
    vocab_vec[:, :x.shape[1]] = x
    if solveroot is not None:
        for i in range(x.shape[1], len(vocab)):
            vocab_vec[:, i] = eval(solveroot[i])
    else:
        vocab_vec = _topdown_hook(hierarchy,
                                  vocab_vec,
                                  x,
                                  vocab,
                                  reduce=reduce)
    vocab_vec += 1e-7
    return vocab_vec


def multiclass_nms_for_cluster(multi_bboxes,
                               multi_scores,
                               labels,
                               nms_thres=0.5):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (np.array): shape (n, #class*4) or (n, 4)
        multi_scores (np.array): shape (n, ),
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = multi_bboxes.max()
    offsets = labels * (max_coordinate + 1)
    bboxes_for_nms = multi_bboxes + offsets[:, None]
    order = np.argsort(multi_scores)[::-1]
    num_box = len(multi_bboxes)
    suppressed = np.zeros(num_box)
    gathered = (np.ones(num_box) * -1).astype(np.int32)
    ious = bbox_overlaps(bboxes_for_nms, bboxes_for_nms)
    for i in range(num_box):
        if suppressed[order[i]]:
            continue
        for j in range(i + 1, num_box):
            if suppressed[order[j]]:
                continue
            iou = ious[order[i], order[j]]
            if iou >= nms_thres:
                suppressed[order[j]] = 1
                gathered[order[j]] = order[i]
    keep = np.where(suppressed == 0)[0]
    return keep, gathered
