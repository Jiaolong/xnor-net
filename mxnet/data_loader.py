import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def load_mnist(data_dir):
    mnist = input_data.read_data_sets(data_dir, one_hot=False)
    x_train = (mnist.train.images > 0).reshape(55000, 1, 28, 28).astype(np.float32)
    y_train = mnist.train.labels[:55000]
    x_test = (mnist.test.images > 0).reshape(10000, 1, 28, 28).astype(np.float32)
    y_test = mnist.test.labels[:10000]
    return (x_train, y_train), (x_test, y_test)
