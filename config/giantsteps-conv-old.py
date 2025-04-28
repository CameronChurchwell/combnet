MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

SAMPLE_RATE = 44_100

HOPSIZE = (SAMPLE_RATE // 5)

N_FFT = 8192

WINDOW_SIZE = 8192

FEATURES = ['spectrogram', 'class']

GIANTSTEPS_KEYS = ['E minor','F minor', 'G minor', 'Db minor', 'C minor', 'Ab major', 'Eb minor', 'G major', 'Bb minor', 'A minor', 'C major', 'D minor', 'Ab minor', 'F major', 'Gb minor', 'B minor', 'Eb major', 'Bb major', 'A major', 'B major', 'D major', 'E major', 'Gb major', 'Db major']

CLASS_MAP = {k: i for i, k in enumerate(GIANTSTEPS_KEYS)}

NUM_CLASSES = len(GIANTSTEPS_KEYS)

NUM_COMB_FILTERS = 24

MODEL_MODULE = 'key_classifiers'

MODEL_CLASS = 'STFTClassifier'

BATCH_SIZE = 16

MEMORY_CACHING = False

import torch
from functools import partial
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.001, momentum=0.9, weight_decay=1e-4)
OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.0001, momentum=0.9, weight_decay=1e-4)

MODEL_KWARGS = {'features': 'quartertones'}

# MODEL_KWARGS = {'n_classes': NUM_CLASSES, 'n_filters': NUM_COMB_FILTERS}
