# Mxnet XNOR-Net #

Proof of concept [XNOR-Net](https://github.com/allenai/XNOR-Net.git) demo on MNIST dataset.

### Data Preparation ###
Download the [mnist dataset](http://yann.lecun.com/exdb/mnist/) and save it to `path_data`.

### Models ###
* mnist_cnn: ordinary convolutional network
* mnist_bwn: binary weight network
* mnist_xnor: xnor-net

### Usage ###
```bash
python main.py --network='mnist_cnn'
```

### Accuracy ###
| Model type | Testing accuracy |
| ------------ | ----------- |
| mnist_cnn | 98.73% |
| mnist_bwn | 98.47% |
| mnist_xnor | 85.40% |
