# ---------------------------------------------------------------
# motif.py
# Set-up time: 2020/5/4 下午4:31
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from mmcv.cnn import kaiming_init
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

from .motif_util import (block_orthogonal, center_x, encode_box_info,
                         get_dropout_mask, obj_edge_vectors, sort_by_score,
                         to_onehot)


class FrequencyBias(nn.Module):
    """The goal of this is to provide a simplified way of computing
    P(predicate.

    | obj1, obj2, img).
    """
    def __init__(self, cfg, statistics, eps=1e-3):
        super(FrequencyBias, self).__init__()
        pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)

        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs * self.num_objs,
                                         self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        ret = self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])
        if ret.isnan().any():
            print('motif: nan')
        return ret

    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2]
        :return:
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:, :, 0].contiguous().view(
            batch_size, num_obj, 1) * pair_prob[:, :, 1].contiguous().view(
                batch_size, 1, num_obj)

        return joint_prob.view(batch_size,
                               num_obj * num_obj) @ self.obj_baseline.weight

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)


class DecoderRNN(nn.Module):
    def __init__(self, config, obj_classes, embed_dim, inputs_dim, hidden_dim,
                 rnn_drop):
        super(DecoderRNN, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.embed_dim = embed_dim

        obj_embed_vecs = obj_edge_vectors(['start'] + self.obj_classes,
                                          wv_dir=self.cfg.glove_dir,
                                          wv_dim=embed_dim)
        self.obj_embed = nn.Embedding(len(self.obj_classes) + 1, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.input_size = self.inputs_dim + self.embed_dim
        self.nms_thresh = 0.5
        self.rnn_drop = rnn_drop

        self.input_linearity = torch.nn.Linear(self.input_size,
                                               6 * self.hidden_size,
                                               bias=True)
        self.state_linearity = torch.nn.Linear(self.hidden_size,
                                               5 * self.hidden_size,
                                               bias=True)
        self.out_obj = nn.Linear(self.hidden_size, len(self.obj_classes))

    def init_weights(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data,
                         [self.hidden_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data,
                         [self.hidden_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.hidden_size:2 *
                                       self.hidden_size].fill_(1.0)

        self.input_linearity.bias.data.fill_(0.0)
        self.input_linearity.bias.data[self.hidden_size:2 *
                                       self.hidden_size].fill_(1.0)

    def lstm_equations(self,
                       timestep_input,
                       previous_state,
                       previous_memory,
                       dropout_mask=None):
        """Does the hairy LSTM math.

        :param timestep_input:
        :param previous_state:
        :param previous_memory:
        :param dropout_mask:
        :return:
        """
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(
            projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
            projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
        forget_gate = torch.sigmoid(
            projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
            projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
        memory_init = torch.tanh(
            projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
            projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
        output_gate = torch.sigmoid(
            projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
            projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        highway_gate = torch.sigmoid(
            projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
            projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
        highway_input_projection = projected_input[:, 5 * self.hidden_size:6 *
                                                   self.hidden_size]
        timestep_output = highway_gate * timestep_output + (
            1 - highway_gate) * highway_input_projection

        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory

    def forward(self,
                inputs,
                initial_state=None,
                labels=None,
                boxes_for_nms=None):
        if not isinstance(inputs, PackedSequence):
            raise ValueError('inputs must be PackedSequence but got %s' %
                             (type(inputs)))

        assert isinstance(inputs, PackedSequence)
        sequence_tensor, batch_lengths, _, _ = inputs
        batch_size = batch_lengths[0]

        # We're just doing an LSTM decoder here so ignore states, etc
        if initial_state is None:
            previous_memory = sequence_tensor.new().resize_(
                batch_size, self.hidden_size).fill_(0)
            previous_state = sequence_tensor.new().resize_(
                batch_size, self.hidden_size).fill_(0)
        else:
            assert len(initial_state) == 2
            previous_memory = initial_state[1].squeeze(0)
            previous_state = initial_state[0].squeeze(0)

        previous_obj_embed = self.obj_embed.weight[0, None].expand(
            batch_size, self.embed_dim)

        if self.rnn_drop > 0.0:
            dropout_mask = get_dropout_mask(self.rnn_drop,
                                            previous_memory.size(),
                                            previous_memory.device)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        out_dists = []
        out_commitments = []

        end_ind = 0
        for i, l_batch in enumerate(batch_lengths):
            start_ind = end_ind
            end_ind = end_ind + l_batch

            if previous_memory.size(0) != l_batch:
                previous_memory = previous_memory[:l_batch]
                previous_state = previous_state[:l_batch]
                previous_obj_embed = previous_obj_embed[:l_batch]
                if dropout_mask is not None:
                    dropout_mask = dropout_mask[:l_batch]

            timestep_input = torch.cat(
                (sequence_tensor[start_ind:end_ind], previous_obj_embed), 1)

            previous_state, previous_memory = self.lstm_equations(
                timestep_input,
                previous_state,
                previous_memory,
                dropout_mask=dropout_mask)

            pred_dist = self.out_obj(previous_state)
            out_dists.append(pred_dist)

            if self.training:
                labels_to_embed = labels[start_ind:end_ind].clone()
                # Whenever labels are 0 set input to be our max prediction
                nonzero_pred = pred_dist[:, 1:].max(1)[1] + 1
                is_bg = (labels_to_embed == 0).nonzero()
                if is_bg.dim() > 0:
                    labels_to_embed[is_bg.squeeze(1)] = nonzero_pred[
                        is_bg.squeeze(1)]
                out_commitments.append(labels_to_embed)
                previous_obj_embed = self.obj_embed(labels_to_embed + 1)
            else:
                # assert l_batch == 1
                out_dist_sample = F.softmax(pred_dist, dim=1)
                best_ind = out_dist_sample[:, 1:].max(1)[1] + 1
                out_commitments.append(best_ind)
                previous_obj_embed = self.obj_embed(best_ind + 1)

        out_commitments = torch.cat(out_commitments, 0)

        return torch.cat(out_dists, 0), out_commitments


class LSTMContext(nn.Module):
    """Modified from neural-motifs to encode contexts for each objects."""
    def __init__(self, config, obj_classes, rel_classes):
        super(LSTMContext, self).__init__()
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
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        obj_embed_vecs = obj_edge_vectors(self.obj_classes,
                                          wv_dir=self.cfg.glove_dir,
                                          wv_dim=self.embed_dim)
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

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.dropout_rate
        self.hidden_dim = self.cfg.hidden_dim
        self.nl_obj = self.cfg.context_object_layer
        self.nl_edge = self.cfg.context_edge_layer
        assert self.nl_obj > 0 and self.nl_edge > 0

        # TODO
        # AlternatingHighwayLSTM is invalid for pytorch 1.0
        self.obj_ctx_rnn = torch.nn.LSTM(
            input_size=self.obj_dim + self.embed_dim + 128,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_obj,
            dropout=self.dropout_rate if self.nl_obj > 1 else 0,
            bidirectional=True)
        self.decoder_rnn = DecoderRNN(self.cfg,
                                      self.obj_classes,
                                      embed_dim=self.embed_dim,
                                      inputs_dim=self.hidden_dim +
                                      self.obj_dim + self.embed_dim + 128,
                                      hidden_dim=self.hidden_dim,
                                      rnn_drop=self.dropout_rate)
        self.edge_ctx_rnn = torch.nn.LSTM(
            input_size=self.embed_dim + self.hidden_dim + self.obj_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_edge,
            dropout=self.dropout_rate if self.nl_edge > 1 else 0,
            bidirectional=True)
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.lin_obj_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

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
        self.decoder_rnn.init_weights()
        for m in self.pos_embed:
            if isinstance(m, nn.Linear):
                kaiming_init(m, distribution='uniform', a=1)
        kaiming_init(self.lin_obj_h, distribution='uniform', a=1)
        kaiming_init(self.lin_edge_h, distribution='uniform', a=1)

    def sort_rois(self, det_result):
        c_x = center_x(det_result)
        # leftright order
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(det_result, scores)

    def obj_ctx(self,
                obj_feats,
                det_result,
                obj_labels=None,
                ctx_average=False):
        """Object context and object classification.

        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param box_priors: [num_obj, 4] boxes. We'll use this for NMS
        :param boxes_per_cls
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self.sort_rois(det_result)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        encoder_rep = self.lin_obj_h(encoder_rep)  # map to hidden_dim

        # untreated decoder input
        batch_size = encoder_rep.shape[0]

        if (not self.training) and self.effect_analysis and ctx_average:
            decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(
                batch_size, -1)
        else:
            decoder_inp = torch.cat((obj_inp_rep, encoder_rep), 1)

        if self.training and self.effect_analysis:
            self.untreated_dcd_feat = self.moving_average(
                self.untreated_dcd_feat, decoder_inp)

        # Decode in order
        if self.mode != 'predcls':
            decoder_inp = PackedSequence(decoder_inp, ls_transposed)
            obj_dists, obj_preds = self.decoder_rnn(
                decoder_inp,  # obj_dists[perm],
                labels=obj_labels[perm] if obj_labels is not None else None)
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_preds, encoder_rep, perm, inv_perm, ls_transposed

    def edge_ctx(self, inp_feats, perm, inv_perm, ls_transposed):
        """Object context and object classification.

        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_reps = self.lin_edge_h(edge_reps)  # map to hidden_dim

        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio
                               ) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, x, det_result, all_average=False, ctx_average=False):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.use_gt_box:  # predcls or sgcls or training, just put obj_labels here
            obj_labels = torch.cat(det_result.labels)
        else:
            obj_labels = None

        if self.use_gt_label:  # predcls
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_dists = torch.cat(det_result.dists, dim=0).detach()
            obj_embed = obj_dists @ self.obj_embed1.weight

        pos_embed = self.pos_embed(encode_box_info(det_result))  # N x 128

        batch_size = x.shape[0]
        if all_average and self.effect_analysis and (
                not self.training):  # TDE: only in test mode
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(
                batch_size, -1)
        else:
            obj_pre_rep = torch.cat((x, obj_embed, pos_embed),
                                    -1)  # N x (1024 + 200 + 128)

        # object level contextual feature
        obj_dists, obj_preds, obj_ctx, perm, inv_perm, ls_transposed = self.obj_ctx(
            obj_pre_rep, det_result, obj_labels, ctx_average=ctx_average)
        # edge level contextual feature
        obj_embed2 = self.obj_embed2(obj_preds.long())

        if (all_average or ctx_average) and self.effect_analysis and (
                not self.training):  # TDE: Testing
            obj_rel_rep = torch.cat((self.untreated_edg_feat.view(
                1, -1).expand(batch_size, -1), obj_ctx),
                                    dim=-1)
        else:
            obj_rel_rep = torch.cat((obj_embed2, x, obj_ctx), -1)

        edge_ctx = self.edge_ctx(obj_rel_rep,
                                 perm=perm,
                                 inv_perm=inv_perm,
                                 ls_transposed=ls_transposed)

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(
                self.untreated_obj_feat, obj_pre_rep)
            self.untreated_edg_feat = self.moving_average(
                self.untreated_edg_feat, torch.cat((obj_embed2, x), -1))

        return obj_dists, obj_preds, edge_ctx, None
