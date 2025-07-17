MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

SAMPLE_RATE = 44_100
N_FFT = 8192

HOPSIZE = (SAMPLE_RATE // 5)

FEATURES = ['audio', 'class']

GIANTSTEPS_KEYS = ['E minor','F minor', 'G minor', 'Db minor', 'C minor', 'Ab major', 'Eb minor', 'G major', 'Bb minor', 'A minor', 'C major', 'D minor', 'Ab minor', 'F major', 'Gb minor', 'B minor', 'Eb major', 'Bb major', 'A major', 'B major', 'D major', 'E major', 'Gb major', 'Db major']

CLASS_MAP = {k: i for i, k in enumerate(GIANTSTEPS_KEYS)}

NUM_CLASSES = len(GIANTSTEPS_KEYS)

MODEL_MODULE = 'key_classifiers'

MODEL_CLASS = 'CombClassifier'

BATCH_SIZE = 8

import torch

PARAM_GROUPS = {
    'main': {'lr': 1e-4},
    'f0': {'lr': 1e-4}
}
OPTIMIZER_FACTORY = torch.optim.Adam

MODEL_KWARGS = {
    'n_filters': 64,
    'comb_kwargs': {
        'min_bin': 20,
        'max_bin': 84,
        'min_freq': 25.95,
        'max_freq': 1046.5
    },
}

F0_INIT_METHOD = 'equal'

STEPS = 100_000

CHECKPOINT_INTERVAL = 10_000

import combnet
import yapecs
if hasattr(combnet, 'defaults'):
    progress_file = Path(__file__).parent / 'giantsteps-comb-trials.progress'

    run = list(range(0, 10))

    # Get grid search parameters for this run
    RANDOM_SEED, = yapecs.grid_search(
        progress_file,
        run)

    CONFIG += f'-{RANDOM_SEED}'