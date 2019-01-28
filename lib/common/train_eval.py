# -*- coding: utf-8 -*-


import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import six
import time

import chainer
from chainer import cuda
from chainer import Variable
from tqdm import tqdm
import random
from sklearn.preprocessing import LabelEncoder

from lib.common import utils
from lib.common.evaluation import evaluate_recall_asym
from lib.common.evaluation import evaluate_recall
from ..datasets import data_provider
from ..models.modified_googlenet import ModifiedGoogLeNet
from ..models.net import Generator, Discriminator
from lib.common.utils import iterate_forward, compute_soft_hard_retrieval, evaluate


def train(main_script_path, func_train_one_batch, param_dict,
          save_distance_matrix=False,):
    script_filename = os.path.splitext(os.path.basename(main_script_path))[0]

    chainer.config.train = False
    device = 0
    xp = chainer.cuda.cupy
    config_parser = six.moves.configparser.ConfigParser()
    config_parser.read('config')
    log_dir_path = os.path.expanduser(config_parser.get('logs', 'dir_path'))

    p = utils.Logger(log_dir_path, **param_dict)  # hyperparameters

    ##########################################################
    # load database
    ##########################################################
    streams = data_provider.get_streams(p.batch_size, dataset=p.dataset,
                                        method=p.method, crop_size=p.crop_size)
    stream_train, stream_train_eval, stream_test = streams
    iter_train = stream_train.get_epoch_iterator()

    ##########################################################
    # construct the model
    ##########################################################
    model = ModifiedGoogLeNet(p.out_dim, p.normalize_output)
    model_gen = Generator()
    model_dis = Discriminator(512, 512)
    if device >= 0:
        model_gen.to_gpu(device)
        model_dis.to_gpu(device)
        model.to_gpu(device)
    model.cleargrads()
    model_gen.cleargrads()
    model_dis.cleargrads()
    xp = model.xp
    
    optimizer_class = getattr(chainer.optimizers, p.optimizer)
    optimizer = optimizer_class(p.learning_rate / 1000)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(p.l2_weight_decay))
    
    optimizer_class = getattr(chainer.optimizers, p.optimizer)
    fea_optimizer = optimizer_class(p.learning_rate)
    fea_optimizer.setup(model)
    fea_optimizer.add_hook(chainer.optimizer.WeightDecay(p.l2_weight_decay))

    # copy version of optimizer for dis and gen
    optimizer_class = getattr(chainer.optimizers, p.optimizer)
    gen_optimizer = optimizer_class(p.learning_rate / 10)
    gen_optimizer.setup(model_gen)
    gen_optimizer.add_hook(chainer.optimizer.WeightDecay(p.l2_weight_decay))

    optimizer_class = getattr(chainer.optimizers, p.optimizer)
    dis_optimizer = optimizer_class(p.learning_rate)
    dis_optimizer.setup(model_dis)
    dis_optimizer.add_hook(chainer.optimizer.WeightDecay(p.l2_weight_decay))

    print(p)
    stop = False
    logger = utils.Logger(log_dir_path)
    logger.soft_test_best = [0]
    time_origin = time.time()
    best_nmi_1 = 0.
    best_f1_1 = 0.
    best_nmi_2 = 0.
    best_f1_2 = 0.
    try:
        for epoch in range(p.num_epochs):
            time_begin = time.time()
            epoch_losses_gen = []
            epoch_losses_dis = []
            loss = 0
            t = tqdm(range(p.num_batches_per_epoch))
            for i in t:
                t.set_description(desc='# {}'.format(epoch))
                with chainer.using_config('train', True):
                    loss_gen, loss_dis = func_train_one_batch(model, model_gen, model_dis, optimizer, fea_optimizer, gen_optimizer, dis_optimizer, p, next(iter_train), epoch)
                epoch_losses_gen.append(loss_gen.data)
                epoch_losses_dis.append(loss_dis.data)
                del loss_gen, loss_dis

            loss_average_gen = cuda.to_cpu(xp.array(
                xp.hstack(epoch_losses_gen).mean()))
            loss_average_dis = cuda.to_cpu(xp.array(
                xp.hstack(epoch_losses_dis).mean()))

            D = [0]
            soft = [0]
            hard = [0]
            retrieval = [0]

            nmi, f1 =evaluate( 
                model, model_dis, stream_test.get_epoch_iterator(), p.distance_type,
                return_distance_matrix=save_distance_matrix,epoch=epoch)
            if nmi > best_nmi_1:
                best_nmi_1 = nmi
                best_f1_1 = f1
                chainer.serializers.save_npz('googlenet.npz',model)
                chainer.serializers.save_npz('model_gen.npz',model_gen)
                chainer.serializers.save_npz('model_dis.npz',model_dis)
            if f1 > best_f1_2:
                best_nmi_2 = nmi
                best_f1_2 = f1

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin

            logger.epoch = epoch
            logger.total_time = total_time
            logger.gen_loss_log.append(loss_average_gen)
            logger.dis_loss_log.append(loss_average_dis)
            logger.train_log.append([soft[0], hard[0], retrieval[0]])
            print("#", epoch)
            print("time: {} ({})".format(epoch_time, total_time))
            print("[train] loss gen:", loss_average_gen)
            print("[train] loss dis:", loss_average_dis)
            print("[test]  nmi:", nmi)
            print("[test]  f1:", f1)
            print("[test]  nmi:", best_nmi_1, "  f1:", best_f1_1, "for max nmi")
            print("[test]  nmi:", best_nmi_2, "  f1:", best_f1_2, "for max f1")
            print(p)
            params = xp.hstack([xp.linalg.norm(param.data)
                                for param in model.params()]).tolist()
            print()

            del D
            del nmi

    except KeyboardInterrupt:
        stop = True

    dir_name = "-".join([p.dataset, script_filename,
                         time.strftime("%Y%m%d%H%M%S"),
                         str(logger.soft_test_best[0])])

    logger.save(dir_name)
    p.save(dir_name)

    print("total epochs: {} ({} [s])".format(logger.epoch, logger.total_time))
    print("best test score (at # {})".format(logger.epoch_best))
    print("[test]  soft:", logger.soft_test_best)
    print("[test]  hard:", logger.hard_test_best)
    print("[test]  retr:", logger.retrieval_test_best)
    print(str(p).replace(', ', '\n'))
    print()
    
    return stop
