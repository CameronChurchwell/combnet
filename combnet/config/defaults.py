import os
from pathlib import Path
import torch

import GPUtil


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'combnet'


###############################################################################
# Data parameters
###############################################################################


# Names of all datasets
DATASETS = ['giantsteps', 'giantsteps_mtg']

# Datasets for evaluation
EVALUATION_DATASETS = ['giantsteps']

FEATURES = ['spectrogram']

INPUT_FEATURES = ['spectrogram']

# SAMPLE_RATE = 16000
SAMPLE_RATE = 44_100

HOPSIZE = (SAMPLE_RATE // 5)

N_FFT = 8192

WINDOW_SIZE = 8192

CLASS_MAP = {}

KEY_MAP = {
    'A# minor': 'Bb minor',
    'C# minor': 'Db minor',
    'D# minor': 'Eb minor',
    'F# minor': 'Gb minor',
    'G# minor': 'Ab minor',
    'A# major': 'Bb major',
    'C# major': 'Db major',
    'D# major': 'Eb major',
    'F# major': 'Gb major',
    'G# major': 'Ab major',
}

GIANTSTEPS_KEYS = ['E minor','F minor', 'G minor', 'Db minor', 'C minor', 'Ab major', 'Eb minor', 'G major', 'Bb minor', 'A minor', 'C major', 'D minor', 'Ab minor', 'F major', 'Gb minor', 'B minor', 'Eb major', 'Bb major', 'A major', 'B major', 'D major', 'E major', 'Gb major', 'Db major']

###############################################################################
# Directories
###############################################################################


# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of preprocessed features
# CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'
CACHE_DIR = Path('/mnt/data3/cameron/combnet') / 'cache'

# Location of datasets on disk
# DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'datasets'
DATA_DIR = Path('/mnt/data3/cameron/combnet') / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = Path(__file__).parent.parent.parent / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = Path(__file__).parent.parent.parent / 'runs'


###############################################################################
# Evaluation parameters
###############################################################################


# Number of steps between tensorboard logging
EVALUATION_INTERVAL = 1_250  # steps

# Number of steps to perform for tensorboard logging
DEFAULT_EVALUATION_STEPS = 4


###############################################################################
# Model parameters
###############################################################################

# model submodule chosen from ['classifiers']
MODEL_MODULE = 'classifiers'

MODEL_CLASS = 'CombClassifier'

MODEL_KWARGS = {}

###############################################################################
# Training parameters
###############################################################################


# Batch size (per gpu)
BATCH_SIZE = 8

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25_000  # steps

# Number of training steps
# STEPS = 10000
STEPS = 100_000

# Number of data loading worker threads
try:
    NUM_WORKERS = int(os.cpu_count() / max(1, len(GPUtil.getGPUs())))
except ValueError:
    NUM_WORKERS = os.cpu_count()

MEMORY_CACHING = False

# Seed for all random number generators
RANDOM_SEED = 1234

OPTIMIZER_FACTORY = torch.optim.Adam

PARAM_GROUPS = None
