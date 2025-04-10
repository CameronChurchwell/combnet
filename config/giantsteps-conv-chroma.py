MODULE = 'combnet'

CONFIG = 'giantsteps-conv-chroma'

SAMPLE_RATE = 44_100

HOPSIZE = (SAMPLE_RATE // 5)

N_FFT = 8192

FEATURES = ['spectrogram', 'class']

GIANTSTEPS_KEYS = ['E minor','F minor', 'G minor', 'Db minor', 'C minor', 'Ab major', 'Eb minor', 'G major', 'Bb minor', 'A minor', 'C major', 'D minor', 'Ab minor', 'F major', 'Gb minor', 'B minor', 'Eb major', 'Bb major', 'A major', 'B major', 'D major', 'E major', 'Gb major', 'Db major']

CLASS_MAP = {k: i for i, k in enumerate(GIANTSTEPS_KEYS)}

NUM_CLASSES = len(GIANTSTEPS_KEYS)

MODEL_MODULE = 'key_classifiers'

MODEL_CLASS = 'STFTClassifier'

BATCH_SIZE = 1

import torch
from functools import partial
OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.001, momentum=0.9, weight_decay=1e-4)
# OPTIMIZER_FACTORY = partial(torch.optim.AdamW, lr=0.000001, weight_decay=1e-4)
# OPTIMIZER_FACTORY = partial(torch.optim.AdamW, lr=0.000005, weight_decay=1e-4)
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.0005, momentum=0.9, weight_decay=1e-4)

MODEL_KWARGS = {'features': 'chroma'}

# MODEL_KWARGS = {'n_classes': NUM_CLASSES, 'n_filters': NUM_COMB_FILTERS}
