# ---------------------------------------------------------------
# sampling.py
# Set-up time: 2020/5/7 下午4:31
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import numpy as np
import numpy.random as npr
import torch
from mmdet.core import bbox_overlaps
from torch.nn import functional as F

# from maskrcnn_benchmark.modeling.box_coder import BoxCoder
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
# from maskrcnn_benchmark.modeling.utils import cat


class RelationSampler(object):
    def __init__(self,
                 type,
                 pos_iou_thr,
                 require_overlap,
                 num_sample_per_gt_rel,
                 num_rel_per_image,
                 pos_fraction,
                 use_gt_box,
                 test_overlap=False,
                 key_sample=False):
        self.type = type
        self.pos_iou_thr = pos_iou_thr
        self.require_overlap = require_overlap
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.num_rel_per_image = num_rel_per_image
        self.pos_fraction = pos_fraction
        self.use_gt_box = use_gt_box
        self.test_overlap = test_overlap
        self.key_sample = key_sample

    def prepare_test_pairs(self, det_result):
        # prepare object pairs for relation prediction
        rel_pair_idxes = []
        device = det_result.bboxes[0].device
        for p in det_result.bboxes:
            n = len(p)
            cand_matrix = torch.ones(
                (n, n), device=device) - torch.eye(n, device=device)
            # mode==sgdet and require_overlap
            # if (not self.use_gt_box) and self.test_overlap:
            if self.test_overlap:
                cand_matrix = cand_matrix.byte() & bbox_overlaps(
                    p[:, :4], p[:, :4]).gt(0).byte()
            idxs = torch.nonzero(cand_matrix).view(-1, 2)
            if len(idxs) > 0:
                rel_pair_idxes.append(idxs)
            else:
                # if there is no candidate pairs, give a placeholder of [[0, 0]]
                rel_pair_idxes.append(
                    torch.zeros((1, 2), dtype=torch.int64, device=device))
        return rel_pair_idxes

    def gtbox_relsample(self, det_result, gt_result):
        assert self.use_gt_box
        num_pos_per_img = int(self.num_rel_per_image * self.pos_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        key_rel_labels = []
        bboxes, labels = det_result.bboxes, det_result.labels
        gt_bboxes, gt_labels, gt_relmaps, gt_rels, gt_keyrels = gt_result.bboxes, gt_result.labels, gt_result.relmaps, \
                                                                gt_result.rels, gt_result.key_rels
        device = bboxes[0].device
        if gt_keyrels is None:
            gt_keyrels = [None] * len(gt_bboxes)
        for img_id, (prp_box, prp_lab, tgt_box, tgt_lab, tgt_rel_matrix,
                     tgt_rel, tgt_keyrel) in enumerate(
                         zip(bboxes, labels, gt_bboxes, gt_labels, gt_relmaps,
                             gt_rels, gt_keyrels)):
            num_prp = prp_box.shape[0]
            assert num_prp == tgt_box.shape[0]
            #tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            tgt_pair_idxs = tgt_rel.long()[:, :2]
            assert tgt_pair_idxs.shape[1] == 2

            # generate the keyrel labels:
            img_keyrel_labels = None
            if tgt_keyrel is not None:
                img_keyrel_labels = torch.zeros(
                    tgt_pair_idxs.shape[0]).long().to(tgt_pair_idxs.device)
                img_keyrel_labels[tgt_keyrel.long()] = 1

            # sort the rel pairs to coordinate with tgt_pair_idxs
            #if tgt_keyrel is not None:
            #perm = torch.from_numpy(np.lexsort((tgt_rel.cpu().numpy()[:, 1], tgt_rel.cpu().numpy()[:, 0]))).to(
            #    device)
            #assert (torch.sum(tgt_rel[perm][:, :2] - tgt_pair_idxs) == 0)
            #tgt_keyrel = tgt_keyrel[perm]

            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = tgt_rel.long()[:, -1].contiguous().view(-1)

            # sym_binary_rels
            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
            rel_sym_binarys.append(binary_rel)

            rel_possibility = torch.ones(
                (num_prp, num_prp), device=device).long() - torch.eye(
                    num_prp, device=device).long()
            rel_possibility[tgt_head_idxs, tgt_tail_idxs] = 0
            rel_possibility[tgt_tail_idxs, tgt_head_idxs] = 0
            tgt_bg_idxs = torch.nonzero(rel_possibility > 0)

            # generate fg bg rel_pairs
            if tgt_pair_idxs.shape[0] > num_pos_per_img:
                perm = torch.randperm(tgt_pair_idxs.shape[0],
                                      device=device)[:num_pos_per_img]
                tgt_pair_idxs = tgt_pair_idxs[perm]
                tgt_rel_labs = tgt_rel_labs[perm]
                if img_keyrel_labels is not None:
                    img_keyrel_labels = img_keyrel_labels[perm]

            num_fg = min(tgt_pair_idxs.shape[0], num_pos_per_img)

            num_bg = self.num_rel_per_image - num_fg
            perm = torch.randperm(tgt_bg_idxs.shape[0], device=device)[:num_bg]
            tgt_bg_idxs = tgt_bg_idxs[perm]

            img_rel_idxs = torch.cat((tgt_pair_idxs, tgt_bg_idxs), dim=0)
            img_rel_labels = torch.cat(
                (tgt_rel_labs.long(),
                 torch.zeros(tgt_bg_idxs.shape[0], device=device).long()),
                dim=0).contiguous().view(-1)
            if img_keyrel_labels is not None:
                img_keyrel_labels = torch.cat(
                    (img_keyrel_labels.long(),
                     torch.ones(tgt_bg_idxs.shape[0], device=device).long() *
                     -1),
                    dim=0).contiguous().view(-1)
                key_rel_labels.append(img_keyrel_labels)

            rel_idx_pairs.append(img_rel_idxs)
            rel_labels.append(img_rel_labels)

        if self.key_sample:
            return rel_labels, rel_idx_pairs, rel_sym_binarys, key_rel_labels
        else:
            return rel_labels, rel_idx_pairs, rel_sym_binarys

    def detect_relsample(self, det_result, gt_result):
        # corresponding to rel_assignments function in neural-motifs
        """
        The input proposals are already processed by subsample function of box_head,
        in this function, we should only care about fg box, and sample corresponding fg/bg relations
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])  contain fields: labels, boxes(5 columns)
            targets (list[BoxList]) contain fields: labels
        """
        if self.type == 'Motif':
            sampling_function = self.motif_rel_fg_bg_sampling
        else:
            raise NotImplementedError
        bboxes, labels = det_result.bboxes, det_result.labels
        gt_bboxes, gt_labels, gt_relmaps, gt_rels, gt_keyrels = gt_result.bboxes, gt_result.labels, gt_result.relmaps,\
                                                                gt_result.rels, gt_result.key_rels
        device = bboxes[0].device
        self.num_pos_per_img = int(self.num_rel_per_image * self.pos_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        key_rel_labels = []
        if gt_keyrels is None:
            gt_keyrels = [None] * len(gt_bboxes)
        for img_id, (prp_box, prp_lab, tgt_box, tgt_lab, tgt_rel_matrix,
                     tgt_rel, tgt_keyrel) in enumerate(
                         zip(bboxes, labels, gt_bboxes, gt_labels, gt_relmaps,
                             gt_rels, gt_keyrels)):
            # IoU matching
            ious = bbox_overlaps(tgt_box, prp_box[:, :4])  # [tgt, prp]
            is_match = (tgt_lab[:, None] == prp_lab[None]) & (
                ious > self.pos_iou_thr)  # [tgt, prp]
            # Proposal self IoU to filter non-overlap
            prp_self_iou = bbox_overlaps(prp_box[:, :4],
                                         prp_box[:, :4])  # [prp, prp]
            if self.require_overlap and (not self.use_gt_box):
                rel_possibility = (prp_self_iou > 0) & (
                    prp_self_iou < 1)  # not self & intersect
            else:
                num_prp = prp_box.shape[0]
                rel_possibility = torch.ones(
                    (num_prp, num_prp), device=device).long() - torch.eye(
                        num_prp, device=device).long()
            # only select relations between fg proposals
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0

            img_rel_triplets, binary_rel = sampling_function(
                device, tgt_rel_matrix, tgt_rel, tgt_keyrel, ious, is_match,
                rel_possibility)
            rel_idx_pairs.append(
                img_rel_triplets[:, :2])  # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2])  # (num_rel, )
            if tgt_keyrel is not None:
                key_rel_labels.append(img_rel_triplets[:, -1])
            rel_sym_binarys.append(binary_rel)

        if self.key_sample:
            return rel_labels, rel_idx_pairs, rel_sym_binarys, key_rel_labels
        else:
            return rel_labels, rel_idx_pairs, rel_sym_binarys

    def motif_rel_fg_bg_sampling(self, device, tgt_rel_matrix, tgt_rel,
                                 tgt_keyrel, ious, is_match, rel_possibility):
        """
        prepare to sample fg relation triplet and bg relation triplet
        tgt_rel_matrix: # [number_target, number_target]
        ious:           # [number_target, num_proposal]
        is_match:       # [number_target, num_proposal]
        rel_possibility:# [num_proposal, num_proposal]
        """
        tgt_pair_idxs = tgt_rel.long()[:, :2]
        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = tgt_rel.long()[:, -1].contiguous().view(-1)

        # # sort the rel pairs to coordinate with tgt_pair_idxs
        # if tgt_keyrel is not None:
        #     perm = torch.from_numpy(np.lexsort((tgt_rel.cpu().numpy()[:, 1]), tgt_rel.cpu().numpy()[:, 0])).to(
        #         device)
        #     assert (torch.sum(tgt_rel[perm][:, :2] - tgt_pair_idxs) == 0)
        #     tgt_keyrel = tgt_keyrel[perm]

        # generate the keyrel labels:
        img_keyrel_labels = None
        if tgt_keyrel is not None:
            img_keyrel_labels = torch.zeros(tgt_pair_idxs.shape[0]).long().to(
                tgt_pair_idxs.device)
            img_keyrel_labels[tgt_keyrel.long()] = 1

        num_tgt_rels = tgt_rel_labs.shape[0]
        # generate binary prp mask
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[
            tgt_head_idxs]  # num_tgt_rel, num_prp (matched prp head)
        binary_prp_tail = is_match[
            tgt_tail_idxs]  # num_tgt_rel, num_prp (matched prp head)
        binary_rel = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []
        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(
                    num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(
                    num_bi_tail, num_bi_head).contiguous()
                # binary rel only consider related or not, so its symmetric
                binary_rel[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = int(tgt_rel_labs[i])
            tgt_key_rel_lab = int(img_keyrel_labels[i]
                                  ) if img_keyrel_labels is not None else None

            # find matching pair in proposals (might be more than one)
            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue
            # all combination pairs
            prp_head_idxs = prp_head_idxs.view(-1, 1).expand(
                num_match_head, num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1, -1).expand(
                num_match_head, num_match_tail).contiguous().view(-1)
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue
            # remove self-pair
            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]
            rel_possibility[prp_head_idxs, prp_tail_idxs] = 0
            # construct corresponding proposal triplets corresponding to i_th gt relation
            fg_labels = torch.tensor([tgt_rel_lab] * prp_tail_idxs.shape[0],
                                     dtype=torch.int64,
                                     device=device).view(-1, 1)
            fg_rel_i = torch.cat((prp_head_idxs.view(
                -1, 1), prp_tail_idxs.view(-1, 1), fg_labels),
                                 dim=-1).to(torch.int64)
            if tgt_key_rel_lab is not None:
                fg_key_labels = torch.tensor([tgt_key_rel_lab] *
                                             prp_tail_idxs.shape[0],
                                             dtype=torch.int64,
                                             device=device).view(-1, 1)
                fg_rel_i = torch.cat((fg_rel_i, fg_key_labels), dim=-1)
            # select if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score
            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = (ious[tgt_head_idx, prp_head_idxs] *
                              ious[tgt_tail_idx, prp_tail_idxs]
                              ).view(-1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = npr.choice(ious_score.shape[0],
                                  p=ious_score,
                                  size=self.num_sample_per_gt_rel,
                                  replace=False)
                fg_rel_i = fg_rel_i[perm]
            if fg_rel_i.shape[0] > 0:
                fg_rel_triplets.append(fg_rel_i)

        # select fg relations
        if len(fg_rel_triplets) == 0:
            col = 4 if self.key_sample else 3
            fg_rel_triplets = torch.zeros((0, col),
                                          dtype=torch.int64,
                                          device=device)
        else:
            fg_rel_triplets = torch.cat(fg_rel_triplets, dim=0).to(torch.int64)
            if fg_rel_triplets.shape[0] > self.num_pos_per_img:
                perm = torch.randperm(fg_rel_triplets.shape[0],
                                      device=device)[:self.num_pos_per_img]
                fg_rel_triplets = fg_rel_triplets[perm]

        # select bg relations
        bg_rel_inds = torch.nonzero(rel_possibility > 0).view(-1, 2)
        bg_rel_labs = torch.zeros(bg_rel_inds.shape[0],
                                  dtype=torch.int64,
                                  device=device)
        bg_rel_triplets = torch.cat((bg_rel_inds, bg_rel_labs.view(-1, 1)),
                                    dim=-1).to(torch.int64)
        if self.key_sample:
            bg_key_labels = torch.tensor(bg_rel_inds.shape[0],
                                         dtype=torch.int64,
                                         device=device).fill_(-1).view(-1, 1)
            bg_rel_triplets = torch.cat((bg_rel_triplets, bg_key_labels),
                                        dim=-1)

        num_neg_per_img = min(
            self.num_rel_per_image - fg_rel_triplets.shape[0],
            bg_rel_triplets.shape[0])
        if bg_rel_triplets.shape[0] > 0:
            perm = torch.randperm(bg_rel_triplets.shape[0],
                                  device=device)[:num_neg_per_img]
            bg_rel_triplets = bg_rel_triplets[perm]
        else:
            bg_rel_triplets = torch.zeros((0, 4 if self.key_sample else 3),
                                          dtype=torch.int64,
                                          device=device)

        # if both fg and bg is none
        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            col = 4 if self.key_sample else 3
            bg_rel_triplets = torch.zeros((1, col),
                                          dtype=torch.int64,
                                          device=device)
            if col == 4:
                bg_rel_triplets[0, -1] = -1

        return torch.cat((fg_rel_triplets, bg_rel_triplets), dim=0), binary_rel
