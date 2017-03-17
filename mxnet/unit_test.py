#! /usr/bin/env python
#--------------------------------------------------
# Simple unit test on binarized convolution layer
#
# Written by Jiaolong Xu
# Date: 03/17/17
# Copyright (c) 2017
#--------------------------------------------------

import numpy as np
import mxnet as mx

from bnn_layers import *

data = mx.sym.Variable('data')
conv1 = mx.sym.Convolution(data=data, num_filter=2, kernel=(2,2), stride=(1,1), pad=(0,0), no_bias=True, name='conv1')
conv2 = mx.sym.Custom(data=data, num_filter=2, kernel=(2,2), stride=(1,1), pad=(0,0), op_type='bin_conv', name='conv1')

dm = mx.nd.array(np.ones([1, 1, 2, 2]).astype(np.float32))
wm = mx.nd.array(np.ones([2, 1, 2, 2]).astype(np.float32))

grad_dm = mx.nd.array(np.ones([1, 1, 2, 2]).astype(np.float32))
grad_wm = mx.nd.array(np.ones([2, 1, 2, 2]).astype(np.float32))

out_grad = np.ones([1, 2, 1, 1]) * 0.5
out_grad_m = mx.nd.array(out_grad)

for conv in [conv1, conv2]:
    e = conv.bind(ctx=mx.cpu(),
            args={'data': dm, 'conv1_weight': wm},
            args_grad=[grad_dm, grad_wm])
    out = e.forward(is_train=True)
    print 'Forward:'
    print(e.arg_dict)
    print(e.output_dict)
    print(e.outputs[0].asnumpy())
    print 'Backward:'
    e.backward(out_grads=out_grad_m)
    print(grad_dm.asnumpy())
    print(grad_wm.asnumpy())
