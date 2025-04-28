import os
from pathlib import Path
import torch
import yapecs
import combnet
from typing import Union, Optional, List, Dict

import GPUtil


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG: str = 'combnet'


###############################################################################
# Data parameters
###############################################################################


# Names of all normal datasets
DATASETS: List[str] = ['giantsteps', 'giantsteps_mtg', 'ptdb', 'mdb', 'maestro']

# Names of all synthetic datasets
SYNTHETIC_DATASETS: List[str] = ['chords']

@yapecs.ComputedProperty(compute_once=False)
def ALL_DATASETS() -> List[str]:
    return combnet.DATASETS + combnet.SYNTHETIC_DATASETS

# Datasets for evaluation
EVALUATION_DATASETS: List[str] = ['giantsteps']

FEATURES: List[str] = ['spectrogram', 'highpass_audio']

INPUT_FEATURES: List[str] = ['spectrogram']

# SAMPLE_RATE = 16000
SAMPLE_RATE: Union[float, int] = 44_100

HOPSIZE: int = (SAMPLE_RATE // 5)

N_FFT: int = 8192

WINDOW_SIZE: int = 8192
# @yapecs.ComputedProperty(compute_once=True)
# def WINDOW_SIZE() -> int:
#     return combnet.N_FFT

KEY_MAP: Dict[str, str] = {
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

GIANTSTEPS_KEYS: List[str] = ['E minor','F minor', 'G minor', 'Db minor', 'C minor', 'Ab major', 'Eb minor', 'G major', 'Bb minor', 'A minor', 'C major', 'D minor', 'Ab minor', 'F major', 'Gb minor', 'B minor', 'Eb major', 'Bb major', 'A major', 'B major', 'D major', 'E major', 'Gb major', 'Db major']

CLASS_MAP: Dict[str, int] = {k: i for i, k in enumerate(GIANTSTEPS_KEYS)}

###############################################################################
# Directories
###############################################################################


ROOT_DIR = Path(__file__).parent.parent.parent

CONFIG_DIR = ROOT_DIR / 'config'

# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of preprocessed features
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'
CACHE_DIR = Path(os.getenv('COMBNET_CACHE_DIR', CACHE_DIR))

# Location of datasets on disk
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'datasets'
DATA_DIR = Path(os.getenv('COMBNET_DATA_DIR', DATA_DIR))

# Location to save evaluation artifacts
EVAL_DIR = Path(__file__).parent.parent.parent / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = Path(__file__).parent.parent.parent / 'runs'


###############################################################################
# Evaluation parameters
###############################################################################


# Number of steps between evaluation tensorboard logging
EVALUATION_INTERVAL = 1_250  # steps

# Number of steps between non-evaluation tensorboard logging
LOG_INTERVAL = 100

# Number of steps to perform for tensorboard logging
DEFAULT_EVALUATION_STEPS = 8

METRICS = ['accuracy', 'loss', 'categorical', 'mirex_weighted']


###############################################################################
# Model parameters
###############################################################################

# model submodule chosen from ['classifiers']
MODEL_MODULE = 'key_classifiers'

MODEL_CLASS = 'CombClassifier'

MODEL_KWARGS = {}

###############################################################################
# Training parameters
###############################################################################


# Batch size (per gpu)
BATCH_SIZE = 8

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25_000  # steps

# Threshold for gradient clipping
GRAD_CLIP_THRESHOLD = 0.5

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

OPTIMIZER_FACTORY: torch.optim.Optimizer = torch.optim.Adam
SCHEDULER_FACTORY: torch.optim.lr_scheduler.LRScheduler = None
SCHEDULER_KWARGS = {}

PARAM_GROUPS = None

# choice from 'equal' and 'random'
F0_INIT_METHOD = 'random'

# TODO document these
FREEZE_POINTS = None
UNFREEZE_POINTS = None
