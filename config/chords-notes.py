from pathlib import Path
import yapecs
globals().update({k:v for k,v in vars(yapecs.import_from_path('chords', Path(__file__).parent / 'chords.py')).items() if not k.startswith('__')})

MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

MODEL_CLASS = 'NotesClassifier'

BATCH_SIZE = 8

FEATURES = ['notes_vector', 'class']

import torch
# from functools import partial
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.001, momentum=0.9)
PARAM_GROUPS = {
    # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 1e-4},
    'main': {'lr': 1e-4},
}
OPTIMIZER_FACTORY = torch.optim.Adam
# OPTIMIZER_FACTORY = partial(torch.optim.AdamW, lr=0.000001, weight_decay=1e-4)
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.0005, momentum=0.9, weight_decay=1e-4)

MODEL_KWARGS = {}

# Number of steps between evaluation tensorboard logging
EVALUATION_INTERVAL = 500  # steps

# Number of training steps
STEPS = 10_000