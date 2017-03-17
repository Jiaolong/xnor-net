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
import logging

from data_loader import load_mnist
from models import mnist_cnn, mnist_bwn, mnist_xnor

os.environ["MXNET_CPU_WORKER_NTHREADS"] = "8"

def main(args):
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
    (x_train, y_train), (x_val, y_val) = load_mnist(args.data_path)
    train_iter = mx.io.NDArrayIter(x_train, y_train, batch_size, shuffle=True)
    test_iter  = mx.io.NDArrayIter(x_val, y_val, batch_size)

    # log
    logging.basicConfig(level=logging.DEBUG)

    # train
    mod = mx.mod.Module(symbol=model, context=mx.cpu())
    mod.fit(train_data=train_iter, eval_data=test_iter, num_epoch=args.num_epoch,
            optimizer_params={'learning_rate':args.learning_rate, 'momentum': args.momentum},
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
    parser.add_argument('--learning_rate', type=float, default=0.1,
            help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
            help='momentum of SGD')
    parser.add_argument('--batch_size', type=int, default=100,
            help='mini-batch size')
    parser.add_argument('--num_epoch', type=int, default=5,
            help='number of training epochs')
    args = parser.parse_args()
    print args

    main(args)
