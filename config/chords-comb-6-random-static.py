from pathlib import Path
import yapecs
globals().update({k:v for k,v in vars(yapecs.import_from_path('chords', Path(__file__).parent / 'chords.py')).items() if not k.startswith('__')})

MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

MODEL_CLASS = 'CombClassifier'

BATCH_SIZE = 8

import torch
# from functools import partial
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.001, momentum=0.9)
PARAM_GROUPS = {
    # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 1e-4},
    'main': {'lr': 5e-5},
    # 'f0': {'lr': 0.1, 'betas': [0.9, 0.999]}
    'f0': {'lr': 0}
    # 'f0': {'lr': 5e-4}
}
OPTIMIZER_FACTORY = torch.optim.Adam
# OPTIMIZER_FACTORY = partial(torch.optim.AdamW, lr=0.000001, weight_decay=1e-4)
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.0005, momentum=0.9, weight_decay=1e-4)

# MODEL_KWARGS = {'n_filters': 16}
MODEL_KWARGS = {
    # 'n_filters': 24,
    'n_filters': 6,
    'comb_kwargs': {
        'min_bin': 20,
        # 'max_bin': 84,
        'max_bin': 50,
        'min_freq': 25.95,
        # 'max_freq': 1046.5
        'max_freq': 261.63,
    }}

# Number of steps between evaluation tensorboard logging
EVALUATION_INTERVAL = 500  # steps

# Number of training steps
STEPS = 50_000
F0_INIT_METHOD = 'random'