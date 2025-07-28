<h1 align="center">Combolutional Neural Networks</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/combnet.svg)](https://pypi.python.org/pypi/combnet)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<!-- [![Downloads](https://pepy.tech/badge/combnet)](https://pepy.tech/project/combnet) -->

</div>

<div align="center">

Training, evaluation, and implementations of Combolutional Layers in PyTorch

[[Paper]]()
</div>


## Table of contents

- [Installation](#installation)
- [Training](#training)
    * [Download](#download)
    * [Preprocess](#preprocess)
    * [Partition](#partition)
    * [Train](#train)
    * [Monitor](#monitor)
    * [Evaluate](#evaluate)
- [References](#references)


## Installation

You can install from PyPI:
`pip install combnet`
or from a local clone:
`pip install -e .`

For full training and evaluation compatibility, you will also need to install FFMPEG version >=4, <7 (version 6 is recommended).


## Training

### Download

`python -m combnet.data.download`

Download and uncompress datasets used for training


### Augmentation

`python -m combnet.data.augment --datasets giantsteps_mtg`

Augment data (pitch shift to other keys)


### Preprocess

`python -m combnet.data.preprocess --datasets giantsteps_mtg giantsteps`

Preprocess datasets


### Partition

`python -m combnet.partition`

Partition datasets. Partitions are saved in `combnet/assets/partitions`.


### Train

`python -m combnet.train --config <config> --gpus <gpus>`

Trains a model according to a given configuration.


### Monitor

Run `tensorboard --logdir runs/`. If you are running training remotely, you
must create a SSH connection with port forwarding to view Tensorboard.
This can be done with `ssh -L 6006:localhost:6006 <user>@<server-ip-address>`.
Then, open `localhost:6006` in your browser.

### Evaluate

```
python -m combnet.evaluate \
    --config <config> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Evaluate a model. `<checkpoint>` is the checkpoint file to evaluate and `<gpu>`
is the GPU index.


## References

