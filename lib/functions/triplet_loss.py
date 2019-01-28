# -*- coding: utf-8 -*-


import numpy as np
import chainer.functions as F



def triplet_loss(a,p,n,alpha=1.0):
    """Lifted struct loss function.

    Args:
        f_a (~chainer.Variable): Feature vectors as anchor examples.
            All examples must be different classes each other.
        f_p (~chainer.Variable): Positive examples corresponding to f_a.
            Each example must be the same class for each example in f_a.
        alpha (~float): The margin parameter.

    Returns:
        ~chainer.Variable: Loss value.

    See: `Deep Metric Learning via Lifted Structured Feature Embedding \
        <http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/\
        Song_Deep_Metric_Learning_CVPR_2016_paper.pdf>`_

    """
    
    distance = F.sum((a - p) ** 2.0, axis = 1) - F.sum((a - n) ** 2.0, axis = 1) +alpha
    return F.average(F.relu(distance)) / 2
