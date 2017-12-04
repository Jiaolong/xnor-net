#! /usr/bin/env python
#--------------------------------------------------
# Binary nueral network layers
#
# Written by Jiaolong Xu
# Date: 03/11/17
# Copyright (c) 2017
#--------------------------------------------------
import os
import mxnet as mx
import numpy as np

try:
    from im2col_cython import im2col_cython, col2im_cython
except ImportError:
  print 'run the following and try again:'
  print 'python setup.py build_ext --inplace'

"""
Binary Activation Layer
"""
class BinaryActivation(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = np.sign(x)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = out_grad[0].asnumpy()
        y[abs(y) > 1] = 0
        y[abs(y) >= 1] = 1
        self.assign(in_grad[0], req[0], mx.nd.array(y))

@mx.operator.register("bin_act")
class BinaryActivationProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(BinaryActivationProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return BinaryActivation()

"""
Binary Convolution Layer
"""
class BinaryConvolution(mx.operator.CustomOp):
    def __init__(self, num_filter, kernel, stride=(1,1), pad=(0,0)):
        self.num_filter = num_filter
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.wbin = None
        self.alpha = None

    def binarize_weight(self, weight):
        """Binarize weight"""
        # alpha[i] = 1/n |W[i]|
        self.wbin = weight
        self.alpha = np.zeros((self.num_filter,), np.float32)
        for i in xrange(self.num_filter):
            self.alpha[i] = (np.sum(abs(weight[i])) * 1.0) / weight[i].size
            self.wbin[i] = np.sign(weight[i]) * self.alpha[i]

    def update_grad(self, wreal, dwbin):
        """
        Update real weight gradient
        """
        dw = dwbin
        for i in xrange(self.num_filter):
            d_sign_w = np.ones_like(wreal[i])
            d_sign_w[abs(wreal[i]) > 1] = 0
            dw[i] = dwbin[i] * (1.0 / wreal[i].size + self.alpha[i] * d_sign_w)
        return dw

    def forward(self, is_train, req, in_data, out_data, aux):
        stride, pad = self.stride, self.pad
        x = in_data[0].asnumpy()
        w = in_data[1].asnumpy()
        x_n, x_d, x_h, x_w = x.shape
        f_n, f_d, f_h, f_w = w.shape
        # check dimensions
        assert (x_w + 2 * pad[0] - f_w) % stride[0] == 0, 'width does not work'
        assert (x_h + 2 * pad[1] - f_h) % stride[1] == 0, 'height does not work'
        # create output
        out_h = (x_h + 2 * pad[0] - f_h) / stride[0] + 1
        out_w = (x_w + 2 * pad[1] - f_w) / stride[1] + 1
        out = np.zeros((x_n, f_n, out_h, out_w), dtype=x.dtype)

        # binarize the weight
        self.binarize_weight(w)

        # convert to colums
        x_cols = im2col_cython(x, f_w, f_h, pad[0], stride[0])
        res = self.wbin.reshape((f_n, -1)).dot(x_cols)
        out = res.reshape(f_n, out_h, out_w, x_n)
        out = out.transpose(3, 0, 1, 2)
        self.assign(out_data[0], req[0], mx.nd.array(out))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        stride, pad = self.stride, self.pad
        x = in_data[0].asnumpy()
        w_real = in_data[1].asnumpy()
        x_n, x_d, x_h, x_w = x.shape
        f_n, _, f_h, f_w = w_real.shape

        # convert to colums
        x_cols = im2col_cython(x, f_w, f_h, pad[0], stride[0])
        dout = out_grad[0].asnumpy()# (x_n, f_n, out_h, out_w)
        dout_ = dout.transpose(1, 2, 3, 0).reshape(f_n, -1)
        dwbin = dout_.dot(x_cols.T).reshape(self.wbin.shape)
        db = np.sum(dout, axis=(0, 2, 3)) # (f_n,)
        dx_cols = self.wbin.reshape(f_n, -1).T.dot(dout_)
        dx = col2im_cython(dx_cols, x_n, x_d, x_h, x_w, f_h, f_w, pad[0], stride[0])

        # update gradient
        dw = self.update_grad(w_real, dwbin)
        self.assign(in_grad[0], req[0], mx.nd.array(dx))
        self.assign(in_grad[1], req[0], mx.nd.array(dw))

@mx.operator.register("bin_conv")
class BinaryConvolutionProp(mx.operator.CustomOpProp):
    def __init__(self, num_filter, kernel, stride=(1,1), pad=(0,0)):
        super(BinaryConvolutionProp, self).__init__(need_top_grad=True)
        self.num_filter = int(num_filter)
        self.kernel = eval(str(kernel))
        self.stride = eval(str(stride))
        self.pad = eval(str(pad))

    def list_arguments(self):
        return ['data', 'weight']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        x_n, x_d, x_h, x_w = in_shape[0]
        x_h += 2 * self.pad[0]
        x_w += 2 * self.pad[1]
        f_n, f_d, f_h, f_w = self.num_filter, x_d, self.kernel[0], self.kernel[1]
        w_shape = [f_n, f_d, f_h, f_w]
        out_h = (x_h - f_h) / self.stride[0] + 1
        out_w = (x_w - f_w) / self.stride[1] + 1
        output_shape = [x_n, f_n, out_h, out_w]
        return [in_shape[0], w_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return BinaryConvolution(self.num_filter, self.kernel, self.stride, self.pad)
