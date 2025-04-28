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
OPTIMIZER_FACTORY = torch.optim.Adam
PARAM_GROUPS = {
    # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 1e-4},
    'main': {'lr': 5e-5},
    # 'f0': {'lr': 0.1, 'betas': [0.9, 0.999]}
    'f0': {'lr': 1e-4}
}

# Import module, but delay updating search params until after configuration
import combnet
if hasattr(combnet, 'defaults'):

    def format_float(f):
        return str(f).replace('.', '_')

    # Lock file to track search progress
    progress_file = Path(__file__).parent / 'giantsteps-comb-n_filters-search.progress'

    # Values that we want to search over
    weight_decay = [1e-4, 0]

    n_filters = list(range(52, 96, 4))

    # Get grid search parameters for this run
    N_FILTERS, = yapecs.grid_search(
        progress_file,
        n_filters,
    )

    MODEL_KWARGS = {
        'n_filters': N_FILTERS,
        'comb_kwargs': {
            'min_bin': 20,
            'max_bin': 84,
            'min_freq': 25.95,
            'max_freq': 1046.5
        },
    }

    print(MODEL_KWARGS)

    CONFIG = f'giantsteps-comb-scaled-equal-{N_FILTERS}'

    F0_INIT_METHOD = 'equal'

    STEPS = 100_000