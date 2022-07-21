# ---------------------------------------------------------------
# vctree.py
# Set-up time: 2020/6/4 上午10:22
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
from mmcv.cnn import xavier_init
from torch import nn
from torch.nn import functional as F

from .motif_util import (encode_box_info, get_dropout_mask, obj_edge_vectors,
                         to_onehot)
from .treelstm_util import (BiTreeLSTM_Backward, BiTreeLSTM_Foreward,
                            MultiLayer_BTreeLSTM, TreeLSTM_IO)
from .vctree_util import (arbForest_to_biForest, generate_forest,
                          get_overlap_info)


class DecoderTreeLSTM(nn.Module):
    def __init__(self,
                 cfg,
                 classes,
                 embed_dim,
                 inputs_dim,
                 hidden_dim,
                 direction='backward',
                 dropout=0.2):
        super(DecoderTreeLSTM, self).__init__()
        """
        Initializes the RNN
        :param embed_dim: Dimension of the embeddings
        :param encoder_hidden_dim: Hidden dim of the encoder, for attention
        :param hidden_dim: Hidden dim of the decoder
        :param vocab_size: Number of words in the vocab
        :param bos_token: To use during decoding (non teacher forcing mode))
        :param bos: beginning of sentence token
        :param unk: unknown token (not used)
        direction = forward | backward
        """
        self.cfg = cfg
        self.classes = classes
        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.nms_thresh = 0.5
        self.dropout = dropout
        # generate embed layer
        embed_vecs = obj_edge_vectors(['start'] + self.classes,
                                      wv_dir=self.cfg.glove_dir,
                                      wv_dim=embed_dim)
        self.obj_embed = nn.Embedding(len(self.classes) + 1, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(embed_vecs, non_blocking=True)
        # generate out layer
        self.out = nn.Linear(self.hidden_size, len(self.classes))

        if direction == 'backward':
            self.input_size = inputs_dim + embed_dim
            self.decoderLSTM = BiTreeLSTM_Backward(self.input_size,
                                                   self.hidden_size,
                                                   is_pass_embed=True,
                                                   embed_layer=self.obj_embed,
                                                   embed_out_layer=self.out)
        elif direction == 'foreward':
            self.input_size = inputs_dim + embed_dim * 2
            self.decoderLSTM = BiTreeLSTM_Foreward(self.input_size,
                                                   self.hidden_size,
                                                   is_pass_embed=True,
                                                   embed_layer=self.obj_embed,
                                                   embed_out_layer=self.out)
        else:
            print('Error Decoder LSTM Direction')

    def forward(self, tree, features, num_obj):
        # generate dropout
        if self.dropout > 0.0:
            dropout_mask = get_dropout_mask(self.dropout,
                                            (1, self.hidden_size),
                                            features.device)
        else:
            dropout_mask = None

        # generate tree lstm input/output class
        h_order = torch.tensor([0] * num_obj, device=features.device)
        lstm_io = TreeLSTM_IO(None, h_order, 0, None, None, dropout_mask)

        self.decoderLSTM(tree, features, lstm_io)

        out_h = lstm_io.hidden[lstm_io.order.long()]
        out_dists = lstm_io.dists[lstm_io.order.long()]
        out_commitments = lstm_io.commitments[lstm_io.order.long()]

        return out_dists, out_commitments


class VCTreeLSTMContext(nn.Module):
    """Modified from neural-motifs to encode contexts for each objects."""
    def __init__(self, config, obj_classes, rel_classes):
        super(VCTreeLSTMContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        in_channels = self.cfg.roi_dim

        self.use_gt_box = self.cfg.use_gt_box
        self.use_gt_label = self.cfg.use_gt_label

        # mode
        if self.cfg.use_gt_box:
            if self.cfg.use_gt_label:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.embed_dim
        obj_embed_vecs = obj_edge_vectors(self.obj_classes,
                                          wv_dir=self.cfg.glove_dir,
                                          wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32),
            nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
        ])

        # overlap embedding
        self.overlap_embed = nn.Sequential(*[
            nn.Linear(6, 128),
            nn.BatchNorm1d(128, momentum=0.001),
            nn.ReLU(inplace=True),
        ])

        # box embed
        self.box_embed = nn.Sequential(*[
            nn.Linear(9, 128),
            nn.BatchNorm1d(128, momentum=0.001),
            nn.ReLU(inplace=True),
        ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.dropout_rate
        self.hidden_dim = self.cfg.hidden_dim
        self.nl_obj = self.cfg.context_object_layer
        self.nl_edge = self.cfg.context_edge_layer
        assert self.nl_obj > 0 and self.nl_edge > 0

        self.obj_reduce = nn.Linear(self.obj_dim, 128)
        self.emb_reduce = nn.Linear(self.embed_dim, 128)
        self.score_pre = nn.Linear(128 * 4, self.hidden_dim)
        self.score_sub = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.score_obj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.vision_prior = nn.Linear(self.hidden_dim * 3, 1)

        self.obj_ctx_rnn = MultiLayer_BTreeLSTM(
            in_dim=self.obj_dim + self.embed_dim + 128,
            out_dim=self.hidden_dim,
            num_layer=self.nl_obj,
            dropout=self.dropout_rate if self.nl_obj > 1 else 0)
        self.decoder_rnn = DecoderTreeLSTM(self.cfg,
                                           self.obj_classes,
                                           embed_dim=self.embed_dim,
                                           inputs_dim=self.hidden_dim +
                                           self.obj_dim + self.embed_dim + 128,
                                           hidden_dim=self.hidden_dim,
                                           dropout=self.dropout_rate)
        self.edge_ctx_rnn = MultiLayer_BTreeLSTM(
            in_dim=self.embed_dim + self.hidden_dim + self.obj_dim,
            out_dim=self.hidden_dim,
            num_layer=self.nl_edge,
            dropout=self.dropout_rate if self.nl_edge > 1 else 0,
        )

        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = self.cfg.causal_effect_analysis

        if self.effect_analysis:
            self.register_buffer(
                'untreated_dcd_feat',
                torch.zeros(self.hidden_dim + self.obj_dim + self.embed_dim +
                            128))
            self.register_buffer(
                'untreated_obj_feat',
                torch.zeros(self.obj_dim + self.embed_dim + 128))
            self.register_buffer('untreated_edg_feat',
                                 torch.zeros(self.embed_dim + self.obj_dim))

    def init_weights(self):
        for module in [self.pos_embed, self.overlap_embed, self.box_embed]:
            for m in module:
                if isinstance(m, nn.Linear):
                    xavier_init(m)
        xavier_init(self.obj_reduce)
        xavier_init(self.emb_reduce)
        xavier_init(self.score_pre)
        xavier_init(self.score_sub)
        xavier_init(self.score_obj)
        xavier_init(self.vision_prior)

    def obj_ctx(self,
                num_objs,
                obj_feats,
                obj_labels=None,
                vc_forest=None,
                ctx_average=False):
        """Object context and object classification.

        :param num_objs:
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param det_result:
        :param vc_forest:
        :param: ctx_average:
        :param obj_labels: [num_obj] the GT labels of the image
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        obj_feats = obj_feats.split(num_objs, dim=0)
        obj_labels = obj_labels.split(
            num_objs, dim=0) if obj_labels is not None else None

        obj_ctxs = []
        obj_preds = []
        obj_dists = []
        for i, (feat, tree) in enumerate(zip(obj_feats, vc_forest)):
            encod_rep = self.obj_ctx_rnn(tree, feat, num_objs[i])
            obj_ctxs.append(encod_rep)
            # Decode in order
            if self.mode != 'predcls':
                if (not self.training
                    ) and self.effect_analysis and ctx_average:
                    decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(
                        encod_rep.shape[0], -1)
                else:
                    decoder_inp = torch.cat((feat, encod_rep), 1)
                if self.training and self.effect_analysis:
                    self.untreated_dcd_feat = self.moving_average(
                        self.untreated_dcd_feat, decoder_inp)
                obj_dist, obj_pred = self.decoder_rnn(tree, decoder_inp,
                                                      num_objs[i])
            else:
                assert obj_labels is not None
                obj_pred = obj_labels[i]
                obj_dist = to_onehot(obj_pred, self.num_obj_classes)
            obj_preds.append(obj_pred)
            obj_dists.append(obj_dist)

        obj_ctxs = torch.cat(obj_ctxs, dim=0)
        obj_preds = torch.cat(obj_preds, dim=0)
        obj_dists = torch.cat(obj_dists, dim=0)
        return obj_ctxs, obj_preds, obj_dists

    def edge_ctx(self, num_objs, obj_feats, forest):
        """Object context and object classification.

        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        inp_feats = obj_feats.split(num_objs, dim=0)

        edge_ctxs = []
        for feat, tree, num_obj in zip(inp_feats, forest, num_objs):
            edge_rep = self.edge_ctx_rnn(tree, feat, num_obj)
            edge_ctxs.append(edge_rep)
        edge_ctxs = torch.cat(edge_ctxs, dim=0)
        return edge_ctxs

    def forward(self, x, det_result, all_average=False, ctx_average=False):
        num_objs = [len(b) for b in det_result.bboxes]
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.use_gt_box:
            obj_labels = torch.cat(det_result.labels)
        else:
            obj_labels = None

        if self.cfg.use_gt_label:
            obj_embed = self.obj_embed1(obj_labels.long())
            obj_dists = F.softmax(to_onehot(obj_labels, self.num_obj_classes))
        else:
            obj_dists = torch.cat(det_result.dists, dim=0).detach()
            obj_embed = obj_dists @ self.obj_embed1.weight

        box_info = encode_box_info(det_result)
        pos_embed = self.pos_embed(box_info)  # N x 128

        batch_size = x.shape[0]
        if all_average and self.effect_analysis and (not self.training):
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(
                batch_size, -1)
        else:
            obj_pre_rep = torch.cat((x, obj_embed, pos_embed), -1)

        # construct VCTree
        box_inp = self.box_embed(box_info)
        pair_inp = self.overlap_embed(get_overlap_info(det_result))
        # 128 + 128 + 128 + 128 = 512
        bi_inp = torch.cat(
            (self.obj_reduce(x.detach()), self.emb_reduce(
                obj_embed.detach()), box_inp, pair_inp), -1)
        bi_preds, vc_scores = self.vctree_score_net(num_objs, bi_inp,
                                                    obj_dists)  # list of N x N
        forest = generate_forest(vc_scores, det_result)
        vc_forest = arbForest_to_biForest(forest)

        # object level contextual feature
        obj_ctxs, obj_preds, obj_dists = self.obj_ctx(num_objs,
                                                      obj_pre_rep,
                                                      obj_labels,
                                                      vc_forest,
                                                      ctx_average=ctx_average)
        # edge level contextual feature
        obj_embed2 = self.obj_embed2(obj_preds.long())

        if (all_average or
                ctx_average) and self.effect_analysis and (not self.training):
            obj_rel_rep = torch.cat((self.untreated_edg_feat.view(
                1, -1).expand(batch_size, -1), obj_ctxs),
                                    dim=-1)
        else:
            obj_rel_rep = torch.cat((obj_embed2, x, obj_ctxs), -1)

        edge_ctx = self.edge_ctx(num_objs, obj_rel_rep, vc_forest)

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(
                self.untreated_obj_feat, obj_pre_rep)
            self.untreated_edg_feat = self.moving_average(
                self.untreated_edg_feat, torch.cat((obj_embed2, x), -1))

        return obj_dists, obj_preds, edge_ctx, bi_preds

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio
                               ) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def vctree_score_net(self, num_objs, roi_feat, roi_dist):
        roi_dist = roi_dist.detach()
        # separate into each image
        roi_feat = F.relu(self.score_pre(roi_feat))  # 512
        sub_feat = F.relu(self.score_sub(roi_feat))  # 512
        obj_feat = F.relu(self.score_obj(roi_feat))  # 512

        sub_feats = sub_feat.split(num_objs, dim=0)
        obj_feats = obj_feat.split(num_objs, dim=0)
        roi_dists = roi_dist.split(num_objs, dim=0)

        bi_preds = []
        vc_scores = []
        for sub, obj, dist in zip(sub_feats, obj_feats, roi_dists):
            # only used to calculate loss
            num_obj = sub.shape[0]
            num_dim = sub.shape[-1]
            sub = sub.view(1, num_obj, num_dim).expand(num_obj, num_obj,
                                                       num_dim)  # N, N, 512
            obj = obj.view(num_obj, 1, num_dim).expand(num_obj, num_obj,
                                                       num_dim)  # N, N, 512
            sub_dist = dist.view(1, num_obj,
                                 -1).expand(num_obj, num_obj,
                                            -1).unsqueeze(2)  # N, N, 1, 151
            obj_dist = dist.view(num_obj, 1,
                                 -1).expand(num_obj, num_obj,
                                            -1).unsqueeze(3)  # N, N, 151, 1
            joint_dist = (sub_dist * obj_dist).view(num_obj, num_obj,
                                                    -1)  # N, N, (151, 151)

            vis_prior = self.vision_prior(
                torch.cat(
                    [sub * obj, sub, obj],
                    #co_prior.unsqueeze(-1)],
                    dim=-1).view(num_obj * num_obj,
                                 -1)).view(num_obj, num_obj)
            joint_pred = F.sigmoid(vis_prior)  #* co_prior

            bi_preds.append(vis_prior)
            vc_scores.append(joint_pred)

        return bi_preds, vc_scores
