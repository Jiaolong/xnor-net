#! /usr/bin/env python
#--------------------------------------------------
# MNIST XNOR-Net Demo
#
# Written by Jiaolong Xu
# Date: 03/11/17
# Copyright (c) 2017
#--------------------------------------------------
import os
import argparse
import mxnet as mx
from mxnet import nd
import logging

from data_loader import get_mnist_iter
from models import mnist_cnn, mnist_bwn, mnist_xnor

os.environ["MXNET_CPU_WORKER_NTHREADS"] = "8"

def main(args, ctx):
    # load model symbol
    if args.network == 'mnist_cnn':
        model = mnist_cnn()
    elif args.network == 'mnist_bwn':
        model = mnist_bwn()
    elif args.network == 'mnist_xnor':
        model = mnist_xnor()
    else:
        raise Exception('Unknown network: ' + args.network)

    # load data
    batch_size = args.batch_size
    train_iter, test_iter = get_mnist_iter(args.data_path, batch_size)
    # train
    mod = mx.mod.Module(symbol=model, context=ctx)
    mod.fit(train_data=train_iter, eval_data=test_iter, num_epoch=args.num_epoch,
            optimizer = 'adam',
            optimizer_params = {
                'learning_rate': args.learning_rate,
                'wd': 0.0,
                'beta1': 0.5
            },
            batch_end_callback=mx.callback.Speedometer(batch_size, 200))

    # evaluate accuracy
    metric = mx.metric.Accuracy()
    test_acc = mod.score(test_iter, metric)
    print ('Testing accuracy: %.2f%%' % (test_acc[0][1] * 100,))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XNOR-Net Demo')
    parser.add_argument('--data_path', type=str, default='../data/MNIST_data/',
            help='path of MNIST dataset')
    parser.add_argument('--network', type=str,
            choices=['mnist_cnn', 'mnist_bwn', 'mnist_xnor'],
            default='mnist_cnn', help='network name')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
            help='learning rate')
    parser.add_argument('--batch_size', type=int, default=100,
            help='mini-batch size')
    parser.add_argument('--num_epoch', type=int, default=5,
            help='number of training epochs')
    args = parser.parse_args()
    print args

    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()

    mx.random.seed(1)
    # log
    logging.basicConfig(level=logging.DEBUG)

    main(args, ctx=ctx)
