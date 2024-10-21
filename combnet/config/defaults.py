import os
from pathlib import Path

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
DATASETS = []

# Datasets for evaluation
EVALUATION_DATASETS = DATASETS


###############################################################################
# Directories
###############################################################################


# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of preprocessed features
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = Path(__file__).parent.parent.parent / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = Path(__file__).parent.parent.parent / 'runs'


###############################################################################
# Evaluation parameters
###############################################################################


# Number of steps between tensorboard logging
EVALUATION_INTERVAL = 2500  # steps

# Number of steps to perform for tensorboard logging
DEFAULT_EVALUATION_STEPS = 16


###############################################################################
# Training parameters
###############################################################################


# Batch size (per gpu)
BATCH_SIZE = 64

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000  # steps

# Number of training steps
STEPS = 300000

# Number of data loading worker threads
try:
    NUM_WORKERS = int(os.cpu_count() / max(1, len(GPUtil.getGPUs())))
except ValueError:
    NUM_WORKERS = os.cpu_count()

# Seed for all random number generators
RANDOM_SEED = 1234
