# ---------------------------------------------------------------
# sgg_eval_util.py
# Set-up time: 2020/5/18 下午9:37
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import numpy as np


def intersect_2d(x1, x2):
    """Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each
    entry is True if those rows match.

    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError('Input arrays must have same #columns')

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res


def argsort_desc(scores):
    """Returns the indices that sort scores descending in a smart way.

    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(
        np.unravel_index(np.argsort(-scores.ravel()), scores.shape))
