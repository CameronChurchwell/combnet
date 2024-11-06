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
DATASETS = ['giantsteps']

# Datasets for evaluation
EVALUATION_DATASETS = DATASETS

FEATURES = ['audio']

SAMPLE_RATE = 16000

HOPSIZE = 160

HOP_LENGTH = 160

N_FFT = 512

CLASS_MAP = {}

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
# EVALUATION_INTERVAL = 2500  # steps
EVALUATION_INTERVAL = 2_500  # steps

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
BATCH_SIZE = 4

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

# Seed for all random number generators
RANDOM_SEED = 1234

OPTIMIZER_FACTORY = torch.optim.Adam
