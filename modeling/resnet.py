
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet-resnet.py

import argparse
from contextlib import ExitStack, contextmanager
from modeling.backbone import freeze_affine_getter
import os

from tensorpack import QueueInput, TFDatasetInput, logger
from tensorpack.callbacks import *
from tensorpack.dataflow import FakeData
from tensorpack.models import *
from tensorpack.tfutils import argscope, SmartInit
from tensorpack.tfutils.varreplace import custom_getter_scope, freeze_variables
from tensorpack.train import SyncMultiGPUTrainerReplicated, TrainConfig, launch_train_with_config
from tensorpack.utils.gpu import get_num_gpu

import tensorflow as tf

from tensorpack.models import BatchNorm, BNReLU, Conv2D, FullyConnected, GlobalAvgPooling, MaxPooling, layer_register
from tensorpack.tfutils.argscope import argscope, get_arg_scope

from config import config as cfg

@layer_register(log_shape=True)
def GroupNorm(x, group=32, gamma_initializer=tf.constant_initializer(1.)):
    """
    More code that reproduces the paper can be found at https://github.com/ppwwyyxx/GroupNorm-reproduce/.
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')

def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


# def get_bn(zero_init=False):
#     """
#     Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
#     """
#     if zero_init:
#         return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
#     else:
#         return lambda x, name=None: BatchNorm('bn', x)

def get_bn(zero_init=False):
    if cfg.BACKBONE.NORM == 'None':
        return lambda x: x
    if cfg.BACKBONE.NORM == 'GN':
        Norm = GroupNorm
        layer_name = 'gn'
    else:
        Norm = BatchNorm
        layer_name = 'bn'
    return lambda x: Norm(layer_name, x, gamma_initializer=tf.zeros_initializer() if zero_init else None)


# ----------------- pre-activation resnet ----------------------
def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def preact_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preact_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preact_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l
# ----------------- pre-activation resnet ----------------------


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def se_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, activation=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, activation=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format in ['NCHW', 'channels_first'] else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnext32x4d_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out * 2, 1, strides=1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out * 2, 3, strides=stride, activation=BNReLU, split=32)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l

@contextmanager
def backbone_scope(freeze):
    """
    Args:
        freeze (bool): whether to freeze all the variables under the scope
    """
    def nonlin(x):
        x = get_bn()(x)
        return tf.nn.relu(x)

    with argscope([Conv2D, MaxPooling, BatchNorm], data_format='channels_first'), \
            argscope(Conv2D, use_bias=False, activation=nonlin,
                     kernel_initializer=tf.variance_scaling_initializer(
                         scale=2.0, mode='fan_out')), \
            ExitStack() as stack:
        if cfg.BACKBONE.NORM in ['FreezeBN', 'SyncBN']:
            if freeze or cfg.BACKBONE.NORM == 'FreezeBN':
                stack.enter_context(argscope(BatchNorm, training=False))
            else:
                stack.enter_context(argscope(
                    BatchNorm, sync_statistics='nccl' if cfg.TRAINER == 'replicated' else 'horovod'))

        if freeze:
            stack.enter_context(freeze_variables(stop_gradient=False, skip_collection=True))
        else:
            # the layers are not completely freezed, but we may want to only freeze the affine
            if cfg.BACKBONE.FREEZE_AFFINE:
                stack.enter_context(custom_getter_scope(freeze_affine_getter))
        yield

def resnet_backbone(image, num_blocks, group_func, block_func):
    # with argscope(Conv2D, use_bias=False,
    #               kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        # Note that TF pads the image by [2, 3] instead of [3, 2].
        # Similar things happen in later stride=2 layers as well.
    freeze_at = cfg.BACKBONE.FREEZE_AT
    with backbone_scope(freeze=freeze_at > 0):
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
    with backbone_scope(freeze=freeze_at > 1):
        c2 = group_func('group0', l, block_func, 64, num_blocks[0], 1)
    with backbone_scope(freeze=False):
        c3 = group_func('group1', c2, block_func, 128, num_blocks[1], 2)
        c4 = group_func('group2', c3, block_func, 256, num_blocks[2], 2)
        c5 = group_func('group3', c4, block_func, 512, num_blocks[3], 2)

        # l = GlobalAvgPooling('gap', l)
        # logits = FullyConnected('linear', l, 1000,
        #                         kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return c2, c3, c4, c5

import sys

thismodule = sys.modules[__name__]

class Resnet_Model(object):
    def __init__(self, depth, data_format, mode='resnet'):
        self.mode = mode
        basicblock = getattr(thismodule,mode + '_basicblock', None)
        bottleneck = getattr(thismodule,mode + '_bottleneck', None)
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]
        self.data_format = data_format
        assert self.block_func is not None, \
            "(mode={}, depth={}) not implemented!".format(mode, depth)

    def output(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(
                image, self.num_blocks,
                preact_group if self.mode == 'preact' else resnet_group, self.block_func)