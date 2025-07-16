from pathlib import Path
import yapecs
globals().update({k:v for k,v in vars(yapecs.import_from_path('notes', Path(__file__).parent / 'notes.py')).items() if not k.startswith('__')})

MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

MODEL_CLASS = 'CombClassifier'

import torch
# from functools import partial
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.001, momentum=0.9)
PARAM_GROUPS = {
    # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 1e-3},
    'main': {'lr': 1e-3},
    # 'f0': {'lr': 0.1, 'betas': [0.9, 0.999]}
    'f0': {'lr': 1e-4}
    # 'f0': {'lr': 5e-4}
}
# OPTIMIZER_FACTORY = partial(torch.optim.AdamW, lr=0.000001, weight_decay=1e-4)
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.0005, momentum=0.9, weight_decay=1e-4)

# MODEL_KWARGS = {'n_filters': 16}

F0_INIT_METHOD = 'equal'

import combnet
import os
# RUNS_DIR = Path(__file__).parent.parent / 'runs.notes.conv'
if hasattr(combnet, 'defaults'):

    channels = [1, 2, 4, 8, 16, 24, 32, 64, 128]
    if os.getenv('COMBNET_NO_SEARCH') is None:
        CHANNELS, = yapecs.grid_search(
            Path(__file__).parent / 'notes-comb-search.progress',
            channels
        )
    else:
        CHANNELS = int(os.getenv('COMBNET_CHANNELS'))

    MODEL_KWARGS = {
    'n_filters': CHANNELS,
    # 'n_filters': 24,
    # 'n_filters': 32,
    'comb_kwargs': {
        # 'min_freq': 20,
        # 'max_freq': 500,
        'min_freq': 200,
        'max_freq': 500,
    }}

    CONFIG += f'-{CHANNELS}'
# space f0 values equally in parameter space (exponentially in frequency space)

# MEMORY_CACHING = True # TODO: remove