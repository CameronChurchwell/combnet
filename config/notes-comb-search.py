from pathlib import Path
import yapecs
globals().update({k:v for k,v in vars(yapecs.import_from_path('notes', Path(__file__).parent / 'notes.py')).items() if not k.startswith('__')})

MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

MODEL_CLASS = 'CombClassifier'

import torch
PARAM_GROUPS = {
    'main': {'lr': 1e-3},
    'f0': {'lr': 1e-4}
}

F0_INIT_METHOD = 'equal'

import combnet
import os

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
    'comb_kwargs': {
        'min_freq': 200,
        'max_freq': 500,
    }}

    CONFIG += f'-{CHANNELS}'