#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import colorama
import os
import chainer.functions as F
from sklearn.model_selection import ParameterSampler
import numpy as np
from lib.common.utils import (
    UniformDistribution, LogUniformDistribution, load_params, lossfun_one_batch)
from lib.common.train_eval import train

colorama.init()



if __name__ == '__main__':
    random_state = None
    num_runs = 100
    save_distance_matrix = False
    param_distributions = dict(
        learning_rate=LogUniformDistribution(low=1e-05, high=1e-04),
    )
    static_params = dict(
        num_epochs=100,
        num_batches_per_epoch=60,
        batch_size=120,
        out_dim=512,
        crop_size=224,
        normalize_output=True,
        normalize_bn=True,
        optimizer='Adam',  # 'Adam' or 'RMSProp'
        distance_type='euclidean',  # 'euclidean' or 'cosine'
        dataset='cars196',  # 'cars196' or 'cub200_2011' or 'products'
        method='triplet',  # sampling method for batch construction
        loss='triplet',
        tradeoff=1.0,
        l2_weight_decay = 1.09216088117e-03,
        alpha = 1.0
    )

    sampler = ParameterSampler(param_distributions, num_runs, random_state)

    for random_params in sampler:
        params = {}
        params.update(random_params)
        params.update(static_params)

        stop = train(__file__, lossfun_one_batch, params,
                     save_distance_matrix)
        if stop:
            break