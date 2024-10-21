<h1 align="center">Deep learning project template</h1>
<div align="center">

<!-- [![PyPI](https://img.shields.io/pypi/v/NAME.svg)](https://pypi.python.org/pypi/NAME) -->
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<!-- [![Downloads](https://pepy.tech/badge/NAME)](https://pepy.tech/project/NAME) -->

</div>


## Table of contents

- [Installation](#installation)
- [Inference](#inference)
    * [Application programming interface](#application-programming-interface)
        * [`NAME.from_text_and_audio`](#NAMEfrom_text_and_audio)
        * [`NAME.from_file`](#NAMEfrom_file)
        * [`NAME.from_file_to_file`](#NAMEfrom_file_to_file)
        * [`NAME.from_files_to_files`](#NAMEfrom_files_to_files)
    * [Command-line interface](#command-line-interface)
- [Training](#training)
    * [Download](#download)
    * [Preprocess](#preprocess)
    * [Partition](#partition)
    * [Train](#train)
    * [Monitor](#monitor)
    * [Evaluate](#evaluate)
- [References](#references)


## Installation

`pip install NAME`


## Inference

```python
import NAME

# TODO - load input
x = None

# Model checkpoint
checkpoint = NAME.DEFAULT_CHECKPOINT

# GPU index
gpu = 0

y = NAME.run(x, checkpoint=checkpoint, gpu=gpu)
```


### Application programming interface

#### `NAME.run`


```
"""

Arguments
    x
        User input
    checkpoint
        The model checkpoint
    gpu
        The GPU index

Returns
    y
        System output
"""
```


#### `NAME.from_file`

```
"""Load from file and process

Arguments
    input_file
        Input file to process
    checkpoint
        The model checkpoint
    gpu : int
        The GPU index

Returns
    y
        System output
"""
```


#### `NAME.from_file_to_file`

```
"""Process file and save to disk

Arguments
    input_file
        Input file to process
    output_file
        Corresponding file to save processed input
    checkpoint
        The model checkpoint
    gpu
        The GPU index
"""
```


#### `NAME.from_files_to_files`

```
"""Process many files and save to disk

Arguments
    input_files
        Input files to process
    output_files
        Corresponding files to save processed input
    checkpoint
        The model checkpoint
    gpu
        The GPU index
"""
```


### Command-line interface

```
python -m NAME
    [-h]
    --input_files INPUT_FILES [INPUT_FILES ...]
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
    [--checkpoint CHECKPOINT]
    [--gpu GPU]

Arguments:
    -h, --help
        show this help message and exit
    --input_files INPUT_FILES [INPUT_FILES ...]
        Input files to process
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
        Corresponding files to save processed inputs
    --checkpoint CHECKPOINT
        The model checkpoint
    --gpu GPU
        The GPU index
```


## Training

### Download

`python -m NAME.data.download`

Download and uncompress datasets used for training


### Preprocess

`python -m NAME.data.preprocess`

Preprocess datasets


### Partition

`python -m NAME.partition`

Partition datasets. Partitions are saved in `NAME/assets/partitions`.


### Train

`python -m NAME.train --config <config> --gpus <gpus>`

Trains a model according to a given configuration. Uses a list of GPU indices
as an argument, and uses distributed data parallelism (DDP) if more than one
index is given. For example, `--gpus 0 3` will train using DDP on GPUs `0`
and `3`.


### Monitor

Run `tensorboard --logdir runs/`. If you are running training remotely, you
must create a SSH connection with port forwarding to view Tensorboard.
This can be done with `ssh -L 6006:localhost:6006 <user>@<server-ip-address>`.
Then, open `localhost:6006` in your browser.

### Evaluate

```
python -m NAME.evaluate \
    --config <config> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Evaluate a model. `<checkpoint>` is the checkpoint file to evaluate and `<gpu>`
is the GPU index.


## References

