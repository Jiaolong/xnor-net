# Implementation of XNOR-Net in mxnet and torch#

Proof of concept [XNOR-Net](https://github.com/allenai/XNOR-Net.git) demo on MNIST dataset.

### Data Preparation ###
Download the [mnist dataset](http://yann.lecun.com/exdb/mnist/) and unpack it to `path_data`.

### Models ###
* mnist_cnn: ordinary convolutional network
* mnist_bwn: binary weight network
* mnist_xnor: xnor-net

### Usage ###
* torch version
```bash
cd torch/xnornet_cpu/
th main.lua -dataDir <path_data> -modelName <mnist_xnor>
```

* mxnet version
```bash
cd mxnet/
python main.py --network='mnist_cnn'
```
### Accuracy ###
* torch version
| Model type | Testing accuracy |
| ------------ | ----------- |
| mnist_cnn | 98.81% |
| mnist_wbn | 98.38% |
| mnist_xnor | 96.64% |

* mxnet version
| Model type | Testing accuracy |
| ------------ | ----------- |
| mnist_cnn | 98.73% |
| mnist_wbn | 97.47% |
| mnist_xnor | 85.40% |
