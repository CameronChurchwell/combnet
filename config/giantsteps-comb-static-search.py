import yapecs
from pathlib import Path

MODULE = 'combnet'

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
# PARAM_GROUPS = {
#     # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
#     'main': {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 1e-4},
#     'f0': {'lr': 0.1, 'momentum': 0.75}
# }
OPTIMIZER_FACTORY = torch.optim.SGD


MODEL_KWARGS = {'n_filters': 24}

# Import module, but delay updating search params until after configuration
import combnet
if hasattr(combnet, 'defaults'):

    def format_float(f):
        return str(f).replace('.', '_')

    # Lock file to track search progress
    progress_file = Path(__file__).parent / 'giantsteps-comb-static-search.progress'

    # Values that we want to search over
    weight_decay = [1e-4, 0]
    momentum = [0.9, 0.5, 0]
    learning_rate = [1, 1e-1, 1e-2]

    # Get grid search parameters for this run
    WEIGHT_DECAY, LEARNING_RATE, MOMENTUM = yapecs.grid_search(
        progress_file,
        weight_decay,
        learning_rate,
        momentum,
    )

    CONFIG = f'giantsteps-comb-static-{format_float(LEARNING_RATE)}-{format_float(MOMENTUM)}-{format_float(WEIGHT_DECAY)}'
    PARAM_GROUPS = {
        # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
        # 'main': {'lr': LEARNING_RATE, 'momentum': 0.9, 'weight_decay': 1e-4},
        'main': {'lr': LEARNING_RATE, 'momentum': MOMENTUM, 'weight_decay': WEIGHT_DECAY},
        'f0': {'lr': 0, 'momentum': 0}
    }