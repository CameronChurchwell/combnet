from pathlib import Path
import yapecs
globals().update({k:v for k,v in vars(yapecs.import_from_path('notes', Path(__file__).parent / 'notes.py')).items() if not k.startswith('__')})

MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

MODEL_CLASS = 'ConvClassifier'

# BATCH_SIZE = 1

import torch
PARAM_GROUPS = {
    'main': {'lr': 1e-3},
}

# MODEL_KWARGS = {'n_channels': 8} # TODO try combnet.HOPSIZE?
MODEL_KWARGS = {'n_channels': 24} # TODO try combnet.HOPSIZE?