from pathlib import Path
import yapecs
globals().update({k:v for k,v in vars(yapecs.import_from_path('notes', Path(__file__).parent / 'notes.py')).items() if not k.startswith('__')})

MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

MODEL_CLASS = 'ConvClassifier'

# BATCH_SIZE = 1

import torch
PARAM_GROUPS = {
    'main': {'lr': 1e-3},
}

import combnet
import os
# RUNS_DIR = Path(__file__).parent.parent / 'runs.notes.conv'
if hasattr(combnet, 'defaults'):

    channels = [1, 2, 4, 8, 16, 24, 32]
    if os.getenv('COMBNET_NO_SEARCH') is None:
        CHANNELS, = yapecs.grid_search(
            Path(__file__).parent / 'notes-conv-search.progress',
            channels
        )
    else:
        CHANNELS = int(os.getenv('COMBNET_CHANNELS'))
    MODEL_KWARGS = {'n_channels': CHANNELS}
    CONFIG += f'-{CHANNELS}'