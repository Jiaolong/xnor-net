#! /usr/bin/env python
#--------------------------------------------------
# MINIST Models
#
# Written by Jiaolong Xu
# Date: 03/13/17
# Copyright (c) 2017
#--------------------------------------------------
import mxnet as mx
from bnn_layers import *

def Block(data, num_filter, kernel, stride=(1,1), pad=(0,0), eps=1e-3, name=None):
    """ CNN block"""
    conv = mx.sym.Convolution(data=data, num_filter=num_filter,
            stride=stride, kernel=kernel, pad=pad, name='conv_%s' % name)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=eps,
            momentum=0.9, name='bn_%s' % name)
    outputs = mx.sym.Activation(data=bn, act_type='relu', name='relu_%s' % name)
    return outputs

def BWBlock(data, num_filter, kernel, stride=(1,1), pad=(0,0), eps=1e-3, name=None):
    """ Binary weight CNN block"""
    conv = mx.sym.Custom(data=data, num_filter=num_filter,
            stride=stride, kernel=kernel, pad=pad, op_type='bin_conv', name='conv_%s' % name)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=eps,
            momentum=0.9, name='bn_%s' % name)
    outputs = mx.sym.Activation(data=bn, act_type='relu', name='relu_%s' % name)
    return outputs

def XNORBlock(data, num_filter, kernel, stride=(1,1), pad=(0,0), eps=1e-4, name=None):
    """ XNOR network block"""
    bn = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps,
            momentum=0.9, name='bn_%s' % name)
    ba = mx.sym.Custom(data=bn, op_type='bin_act', name='binAct_%s' % name)
    outputs = mx.sym.Custom(data=ba, num_filter=num_filter,
            stride=stride, kernel=kernel, pad=pad, op_type='bin_conv', name='conv_%s' % name)
    return outputs

def mnist_cnn():
    """ordinary CNN network"""
    data = mx.sym.Variable('data')
    blk1 = Block(data=data, num_filter=32, kernel=(5,5), name='blk_1') # 28 x28 -> 24 x 24
    max1 = mx.sym.Pooling(data=blk1, kernel=(3,3), stride=(3,3),
            pool_type='max', name='max_pool_1') # 24 x 24 -> 8 x 8
    blk2 = Block(data=max1, num_filter=64, kernel=(5,5), name='blk_2') # 8 x 8 -> 4 x 4
    max2 = mx.sym.Pooling(data=blk2, kernel=(2,2), stride=(2,2),
            pool_type='max', name='max_pool_2') # 4 x 4 -> 2 x 2
    blk3 = Block(data=max2, num_filter=200, kernel=(2,2), name='blk_3') # 2 x 2 -> 1 x 1
    conv = mx.sym.Convolution(data=blk3, num_filter=10, kernel=(1,1), name='conv_1x1') # 1x1 conv
    flatten = mx.sym.Flatten(data=conv, name='flatten1') # flatten
    softmax = mx.sym.SoftmaxOutput(data=flatten, name='softmax')
    return softmax

def mnist_bwn():
    """Binary weight network"""
    data = mx.sym.Variable('data')
    blk1 = Block(data=data, num_filter=32, kernel=(5,5), name='blk_1') # 28 x28 -> 24 x 24
    max1 = mx.sym.Pooling(data=blk1, kernel=(3,3), stride=(3,3),
            pool_type='max', name='max_pool_1') # 24 x 24 -> 8 x 8

    blk2 = BWBlock(data=max1, num_filter=64, kernel=(5,5), name='blk_2') # 8 x 8 -> 4 x 4
    max2 = mx.sym.Pooling(data=blk2, kernel=(2,2), stride=(2,2),
            pool_type='max', name='max_pool_2') # 4 x 4 -> 2 x 2
    blk3 = BWBlock(data=max2, num_filter=200, kernel=(2,2), name='blk_3') # 2 x 2 -> 1 x 1

    conv = mx.sym.Convolution(data=blk3, num_filter=10, kernel=(1,1), name='conv_1x1') # 1x1 conv
    flatten = mx.sym.Flatten(data=conv, name='flatten1') # flatten
    softmax = mx.sym.SoftmaxOutput(data=flatten, name='softmax')
    return softmax

def mnist_xnor():
    """xnor network"""
    data = mx.sym.Variable('data')
    blk1 = Block(data=data, num_filter=32, kernel=(5,5), eps=1e-5, name='blk_1') # 28 x28 -> 24 x 24
    max1 = mx.sym.Pooling(data=blk1, kernel=(3,3), stride=(3,3),
            pool_type='max', name='max_pool_1') # 24 x 24 -> 8 x 8

    blk2 = XNORBlock(data=max1, num_filter=64, kernel=(5,5), name='blk_2') # 8 x 8 -> 4 x 4
    max2 = mx.sym.Pooling(data=blk2, kernel=(2,2), stride=(2,2),
            pool_type='max', name='max_pool_2') # 4 x 4 -> 2 x 2
    blk3 = XNORBlock(data=max2, num_filter=200, kernel=(2,2), name='blk_3') # 2 x 2 -> 1 x 1

    bn1  = mx.sym.BatchNorm(data=blk3, fix_gamma=False, eps=1e-3,
            momentum=0.9, name='bn_1')
    ac1  = mx.sym.Activation(data=bn1, act_type='relu', name='relu_1')

    conv = mx.sym.Convolution(data=ac1, num_filter=10, kernel=(1,1), name='conv_1x1') # 1x1 conv
    flatten = mx.sym.Flatten(data=conv, name='flatten1') # flatten
    softmax = mx.sym.SoftmaxOutput(data=flatten, name='softmax')
    return softmax
