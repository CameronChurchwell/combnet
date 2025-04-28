MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

SAMPLE_RATE = 44_100
N_FFT = 8192
# SAMPLE_RATE = 16_000

HOPSIZE = (SAMPLE_RATE // 5)

FEATURES = ['audio', 'class']

GIANTSTEPS_KEYS = ['E minor','F minor', 'G minor', 'Db minor', 'C minor', 'Ab major', 'Eb minor', 'G major', 'Bb minor', 'A minor', 'C major', 'D minor', 'Ab minor', 'F major', 'Gb minor', 'B minor', 'Eb major', 'Bb major', 'A major', 'B major', 'D major', 'E major', 'Gb major', 'Db major']

CLASS_MAP = {k: i for i, k in enumerate(GIANTSTEPS_KEYS)}

NUM_CLASSES = len(GIANTSTEPS_KEYS)

MODEL_MODULE = 'key_classifiers'

MODEL_CLASS = 'CombClassifier'

BATCH_SIZE = 8

import torch
# from functools import partial
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.001, momentum=0.9)
# PARAM_GROUPS = {
#     # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
#     'main': {'lr': 1e-4},
#     # 'f0': {'lr': 0.1, 'momentum': 0.75}
#     'f0': {'lr': 1e-2},
# }
PARAM_GROUPS = {
    # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 1e-4},
    'main': {'lr': 5e-5},
    # 'f0': {'lr': 0.1, 'betas': [0.9, 0.999]}
    'f0': {'lr': 1e-4}
}
OPTIMIZER_FACTORY = torch.optim.Adam
# OPTIMIZER_FACTORY = partial(torch.optim.Adam, lr=0.000001, weight_decay=1e-4)
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.0005, momentum=0.9, weight_decay=1e-4)

MODEL_KWARGS = {
    'n_filters': 32,
    'n_conv_layers': 6,
    'comb_kwargs': {
        'min_bin': 20,
        'max_bin': 84,
        'min_freq': 25.95,
        'max_freq': 1046.5
    }
}