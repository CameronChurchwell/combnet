from pathlib import Path
import yapecs
globals().update({k:v for k,v in vars(yapecs.import_from_path('notes', Path(__file__).parent / 'notes.py')).items() if not k.startswith('__')})

MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

MODEL_CLASS = 'CombClassifier'

PARAM_GROUPS = {
    'main': {'lr': 1e-3},
    'f0': {'lr': 1e-4}
}


MODEL_KWARGS = {
    'n_filters': 16,
    'comb_kwargs': {
        'min_freq': 200,
        'max_freq': 500,
    }}


F0_INIT_METHOD = 'equal'