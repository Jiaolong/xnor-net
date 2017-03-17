# XNOR-Net #

Proof of concept [XNOR-Net](https://github.com/allenai/XNOR-Net.git) demo on MNIST dataset.

### Data Preparation ###
Download the [mnist dataset](http://yann.lecun.com/exdb/mnist/) and unpack it to `path_data`.

### Models ###
* mnist_cnn: ordinary convolutional network
* mnist_bwn: binary weight network
* mnist_xnor: xnor-net

### Usage ###
```bash
th main.lua -dataDir <path_data> -modelName <mnist_xnor>
```

### Accuracy ###
| Model type | Testing accuracy |
| ------------ | ----------- |
| mnist_cnn | 98.81% |
| mnist_wbn | 98.38% |
| mnist_xnor | 96.64% |
