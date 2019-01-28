# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    def __init__(self, out_dim=512):
        super(Generator, self).__init__()
        self.out_dim = out_dim

        with self.init_scope():
            w = chainer.initializers.Normal(0.02)
            self.l0 = L.Linear(self.out_dim*3, self.out_dim,
                               initialW=w)

            self.l1 = L.Linear(self.out_dim, self.out_dim,
                               initialW=w)
    def __call__(self, z):
        h = F.relu(self.l0(z))
        h = F.relu(self.l1(h))
        return h


class Discriminator(chainer.Chain):
    def __init__(self, in_dim, out_dim):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        w = chainer.initializers.Identity()
        with self.init_scope():
            self.l0 = L.Linear(self.in_dim, self.out_dim,
                               initialW=w)

            self.l1 = L.Linear(self.out_dim, self.out_dim,
                               initialW=w)

    def __call__(self, x):
        h = F.relu(self.l0(x))
        return self.l1(h)