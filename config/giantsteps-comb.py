MODULE = 'combnet'

CONFIG = 'giantsteps-comb'

# SAMPLE_RATE = 44_100
SAMPLE_RATE = 16_000

HOPSIZE = (SAMPLE_RATE // 5)

FEATURES = ['audio', 'class']

GIANTSTEPS_KEYS = ['E minor','F minor', 'G minor', 'Db minor', 'C minor', 'Ab major', 'Eb minor', 'G major', 'Bb minor', 'A minor', 'C major', 'D minor', 'Ab minor', 'F major', 'Gb minor', 'B minor', 'Eb major', 'Bb major', 'A major', 'B major', 'D major', 'E major', 'Gb major', 'Db major']

CLASS_MAP = {k: i for i, k in enumerate(GIANTSTEPS_KEYS)}

NUM_CLASSES = len(GIANTSTEPS_KEYS)

MODEL_MODULE = 'classifiers'

MODEL_CLASS = 'CombClassifier'

BATCH_SIZE = 8

import torch
from functools import partial
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.001, momentum=0.9)
PARAM_GROUPS = {
    'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
    'f0': {'lr': 10000, 'momentum': 0.9}
}
OPTIMIZER_FACTORY = torch.optim.SGD
# OPTIMIZER_FACTORY = partial(torch.optim.AdamW, lr=0.000001, weight_decay=1e-4)
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.0005, momentum=0.9, weight_decay=1e-4)


MODEL_KWARGS = {'n_filters': 12}