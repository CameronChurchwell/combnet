from pathlib import Path
import yapecs
globals().update(vars(yapecs.import_from_path('chords', Path(__file__).parent / 'chords.py')))

MODULE = 'combnet'

CONFIG = 'chords-comb-3-static'

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
    'f0': {'lr': 0., 'betas': [0.9, 0.999]}
}
OPTIMIZER_FACTORY = torch.optim.Adam
# OPTIMIZER_FACTORY = partial(torch.optim.AdamW, lr=0.000001, weight_decay=1e-4)
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.0005, momentum=0.9, weight_decay=1e-4)

# MODEL_KWARGS = {'n_filters': 16}
MODEL_KWARGS = {'n_filters': 3}

# Number of steps between evaluation tensorboard logging
EVALUATION_INTERVAL = 500  # steps

# Number of training steps
STEPS = 100_000