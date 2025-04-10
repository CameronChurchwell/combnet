from pathlib import Path
import os
import yapecs
MODULE = 'combnet'

base_config_path = Path(__file__).parent / 'chords.py'
base_config_path = os.getenv(f'{MODULE.upper()}_BASE_CONFIG', base_config_path)
globals().update(vars(yapecs.import_from_path('chords', base_config_path)))


MODEL_CLASS = 'ConvClassifier'

BATCH_SIZE = 8

RUNS_DIR = Path(__file__).parent.parent / 'runs.conv'

import torch
import combnet
if hasattr(combnet, 'defaults'):

    def format_float(f):
        return str(f).replace('.', '_')

    # learning_rate = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    learning_rate = [1e-2, 5e-3, 1e-3]
    # kernel_size = [32, 48, 64, 128]
    kernel_size = [192, 256]
    # n_channels = [16, 32, 48]
    n_channels = [4]

    if os.getenv('COMBNET_NO_SEARCH') is None:
        LEARNING_RATE, KERNEL_SIZE, N_CHANNELS = yapecs.grid_search(
            Path(__file__).parent / 'chords-conv-search.progress',
            learning_rate,
            kernel_size,
            n_channels
        )
    else:
        LEARNING_RATE = float(os.getenv('COMBNET_LEARNING_RATE'))
        KERNEL_SIZE = int(os.getenv('COMBNET_KERNEL_SIZE'))
        N_CHANNELS = int(os.getenv('COMBNET_N_CHANNELS'))

    CONFIG = f'chords-conv-{format_float(LEARNING_RATE)}-{KERNEL_SIZE}-{N_CHANNELS}'

    # from functools import partial
    # OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.001, momentum=0.9)
    PARAM_GROUPS = {
        # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
        # 'main': {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 1e-4},
        # 'main': {'lr': 5e-5},
        'main': {'lr': LEARNING_RATE},
    }
    OPTIMIZER_FACTORY = torch.optim.Adam
    # OPTIMIZER_FACTORY = partial(torch.optim.AdamW, lr=0.000001, weight_decay=1e-4)
    # OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.0005, momentum=0.9, weight_decay=1e-4)

    MODEL_KWARGS = {'kernel_size': KERNEL_SIZE, 'n_channels': N_CHANNELS, 'stride': 4} # TODO try combnet.HOPSIZE?

# Number of steps between evaluation tensorboard logging
EVALUATION_INTERVAL = 500  # steps

CHECKPOINT_INTERVAL = 5_000

# Number of training steps
STEPS = 10_000