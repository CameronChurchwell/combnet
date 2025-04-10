from pathlib import Path
import yapecs
globals().update(vars(yapecs.import_from_path('chords', Path(__file__).parent / 'chords.py')))

MODULE = 'combnet'

CONFIG = 'chords-conv'

MODEL_CLASS = 'ConvClassifier'

BATCH_SIZE = 8

import torch
# from functools import partial
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.001, momentum=0.9)
PARAM_GROUPS = {
    # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 5e-5},
    'main': {'lr': 1e-3},
}
OPTIMIZER_FACTORY = torch.optim.Adam
# OPTIMIZER_FACTORY = partial(torch.optim.AdamW, lr=0.000001, weight_decay=1e-4)
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.0005, momentum=0.9, weight_decay=1e-4)

MODEL_KWARGS = {'kernel_size': 256, 'n_channels': 8, 'stride': 4} # TODO try combnet.HOPSIZE?

# Number of steps between evaluation tensorboard logging
EVALUATION_INTERVAL = 500  # steps

CHECKPOINT_INTERVAL = 5_000 # steps

# Number of training steps
STEPS = 10_000 # steps