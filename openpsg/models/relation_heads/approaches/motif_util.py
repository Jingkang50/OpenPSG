# ---------------------------------------------------------------
# motif_util.py
# Set-up time: 2020/5/4 下午4:36
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import array
import itertools
import os
import sys
import zipfile

import numpy as np
import six
import torch
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm


def normalize_sigmoid_logits(orig_logits):
    orig_logits = torch.sigmoid(orig_logits)
    orig_logits = orig_logits / (orig_logits.sum(1).unsqueeze(-1) + 1e-12)
    return orig_logits


def generate_attributes_target(attributes, device, max_num_attri,
                               num_attri_cat):
    """from list of attribute indexes to [1,0,1,0,0,1] form."""
    assert max_num_attri == attributes.shape[1]
    num_obj = attributes.shape[0]

    with_attri_idx = (attributes.sum(-1) > 0).long()
    attribute_targets = torch.zeros((num_obj, num_attri_cat),
                                    device=device).float()

    for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
        for k in range(max_num_attri):
            att_id = int(attributes[idx, k])
            if att_id == 0:
                break
            else:
                attribute_targets[idx, att_id] = 1
    return attribute_targets, with_attri_idx


def transpose_packed_sequence_inds(lengths):
    """Get a TxB indices from sorted lengths.

    Fetch new_inds, split by new_lens, padding to max(new_lens), and stack.
    Returns:
        new_inds (np.array) [sum(lengths), ]
        new_lens (list(np.array)): number of elements of each time step,
        descending
    """
    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer + 1)].copy())
        cum_add[:(length_pointer + 1)] += 1
        new_lens.append(length_pointer + 1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens


def sort_by_score(infostruct, scores):
    """We'll sort everything scorewise from Hi->low, BUT we need to keep images
    together and sort LSTM from l.

    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_rois = [len(b) for b in infostruct.bboxes]
    num_im = len(num_rois)

    scores = scores.split(num_rois, dim=0)

    ordered_scores = []
    for i, (score, num_roi) in enumerate(zip(scores, num_rois)):
        ordered_scores.append(score + 2.0 * float(num_roi * 2 * num_im - i))
    ordered_scores = torch.cat(ordered_scores, dim=0)
    _, perm = torch.sort(ordered_scores, 0, descending=True)

    num_rois = sorted(num_rois, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(
        num_rois)  # move it to TxB form
    inds = torch.LongTensor(inds).to(scores[0].device)
    ls_transposed = torch.LongTensor(ls_transposed)

    perm = perm[inds]  # (batch_num_box, )
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed


def to_onehot(vec, num_classes, fill=1000):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill

    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return:
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    arange_inds = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=arange_inds)

    onehot_result.view(-1)[vec.long() + num_classes * arange_inds] = fill
    return onehot_result


def get_dropout_mask(dropout_probability, tensor_shape, device):
    """once get, it is fixed all the time."""
    binary_mask = (torch.rand(tensor_shape) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().to(device).div(1.0 -
                                                      dropout_probability)
    return dropout_mask


def center_x(infostruct):
    boxes = torch.cat(infostruct.bboxes, dim=0)
    c_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    return c_x.view(-1)


def encode_box_info(infostruct):
    """encode proposed box information (x1, y1, x2, y2) to (cx/wid, cy/hei,
    w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)"""
    bboxes, img_shapes = infostruct.bboxes, infostruct.img_shape
    boxes_info = []
    for bbox, img_shape in zip(bboxes, img_shapes):
        wid = img_shape[1]
        hei = img_shape[0]
        wh = bbox[:, 2:4] - bbox[:, 0:2] + 1.0
        xy = bbox[:, 0:2] + 0.5 * wh
        w, h = wh[:, 0], wh[:, 1]
        x, y = xy[:, 0], xy[:, 1]
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        assert wid * hei != 0
        info = torch.stack([
            w / wid, h / hei, x / wid, y / hei, x1 / wid, y1 / hei, x2 / wid,
            y2 / hei, w * h / (wid * hei)
        ],
                           dim=-1).view(-1, 9)
        boxes_info.append(info)

    return torch.cat(boxes_info, dim=0)


def obj_edge_vectors(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0, 1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word
            lw_token = sorted(token.split(' '),
                              key=lambda x: len(x),
                              reverse=True)[0]
            print('{} -> {} '.format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print('fail on {}'.format(token))

    return vectors


def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B':
        'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)

    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        try:
            return torch.load(fname_pt, map_location=torch.device('cpu'))
        except Exception as e:
            print('Error loading the model from {}{}'.format(fname_pt, str(e)))
            sys.exit(-1)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, 'r') as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(
                range(len(cm)),
                desc='loading word vectors from {}'.format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret


def reporthook(t):
    """https://github.com/tqdm/tqdm."""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def block_orthogonal(tensor, split_sizes, gain=1.0):
    """
    An initializer which allows initializing model parameters in "blocks".
    This is helpful in the case of recurrent models which use multiple
    gates applied to linear projections, which can be computed efficiently
    if they are concatenated together.
    However, they are separate parameters which should be initialized
    independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError(
            'tensor dimensions must be divisible by their respective '
            'split_sizes. Found size: {} and split_sizes: {}'.format(
                sizes, split_sizes))
    indexes = [
        list(range(0, max_size, split))
        for max_size, split in zip(sizes, split_sizes)
    ]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([
            slice(start_index, start_index + step)
            for start_index, step in index_and_step_tuples
        ])

        # not initialize empty things to 0s because THAT SOUNDS REALLY BAD
        assert len(block_slice) == 2
        sizes = [x.stop - x.start for x in block_slice]
        tensor_copy = tensor.new(max(sizes), max(sizes))
        torch.nn.init.orthogonal(tensor_copy, gain=gain)
        tensor[block_slice] = tensor_copy[0:sizes[0], 0:sizes[1]]
