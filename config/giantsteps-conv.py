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

MODEL_CLASS = 'ConvClassifier'

BATCH_SIZE = 8

import torch

from functools import partial
OPTIMIZER_FACTORY = partial(torch.optim.Adam, lr=0.01)

MODEL_KWARGS = {'kernel_size': SAMPLE_RATE//5, 'n_channels': 64, 'stride': SAMPLE_RATE//5}