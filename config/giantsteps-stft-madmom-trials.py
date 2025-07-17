MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

SAMPLE_RATE = 44_100

HOPSIZE = (SAMPLE_RATE // 5)

N_FFT = 8192

FEATURES = ['spectrogram', 'class']

GIANTSTEPS_KEYS = ['E minor','F minor', 'G minor', 'Db minor', 'C minor', 'Ab major', 'Eb minor', 'G major', 'Bb minor', 'A minor', 'C major', 'D minor', 'Ab minor', 'F major', 'Gb minor', 'B minor', 'Eb major', 'Bb major', 'A major', 'B major', 'D major', 'E major', 'Gb major', 'Db major']

CLASS_MAP = {k: i for i, k in enumerate(GIANTSTEPS_KEYS)}

NUM_CLASSES = len(GIANTSTEPS_KEYS)

MODEL_MODULE = 'key_classifiers'

MODEL_CLASS = 'STFTClassifier'

STEPS = 100_000

EVALUATION_INTERVAL = 1250

BATCH_SIZE = 8

import torch
from functools import partial
OPTIMIZER_FACTORY = partial(torch.optim.Adam, lr=0.0005)

MODEL_KWARGS = {'features': 'madmom'}

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