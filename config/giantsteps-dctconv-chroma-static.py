MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

SAMPLE_RATE = 44_100

HOPSIZE = (SAMPLE_RATE // 5)

N_FFT = 8192

FEATURES = ['audio', 'class']

GIANTSTEPS_KEYS = ['E minor','F minor', 'G minor', 'Db minor', 'C minor', 'Ab major', 'Eb minor', 'G major', 'Bb minor', 'A minor', 'C major', 'D minor', 'Ab minor', 'F major', 'Gb minor', 'B minor', 'Eb major', 'Bb major', 'A major', 'B major', 'D major', 'E major', 'Gb major', 'Db major']

CLASS_MAP = {k: i for i, k in enumerate(GIANTSTEPS_KEYS)}

NUM_CLASSES = len(GIANTSTEPS_KEYS)

MODEL_MODULE = 'key_classifiers'

MODEL_CLASS = 'DCTConvClassifier'

# STEPS = 100_000 * 16

# EVALUATION_INTERVAL = 1250 * 16

# BATCH_SIZE = 1

STEPS = 100_000

EVALUATION_INTERVAL = 1250

BATCH_SIZE = 8

import torch
OPTIMIZER_FACTORY = torch.optim.Adam

MODEL_KWARGS = {'features': 'chroma', 'init_dct': True, 'init_filters': True}

PARAM_GROUPS = {
    'main': {'lr': 0.0005},
    'dct': {'lr': 0},
    'filters': {'lr': 0},
}