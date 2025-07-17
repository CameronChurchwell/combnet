from pathlib import Path
import yapecs
globals().update({k:v for k,v in vars(yapecs.import_from_path('timit-frame', Path(__file__).parent / 'timit-frame.py')).items() if not k.startswith('__')})

MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

MODEL_CLASS = 'SincNet'

PARAM_GROUPS = {
    'main': {'lr': 0.001},
}

RANDOM_SEED = 24