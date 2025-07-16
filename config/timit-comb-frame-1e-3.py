from pathlib import Path
import yapecs
globals().update({k:v for k,v in vars(yapecs.import_from_path('timit-frame', Path(__file__).parent / 'timit-frame.py')).items() if not k.startswith('__')})

MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

MODEL_CLASS = 'CombClassifier'

PARAM_GROUPS = {
    # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 1e-3},
    'main': {'lr': 0.001},
    # 'f0': {'lr': 0.1, 'betas': [0.9, 0.999]}
    'f0': {'lr': 1e-3}
    # 'f0': {'lr': 5e-4}
}
MODEL_KWARGS = {
    'n_filters': 80,
    'comb_kwargs': {
        'min_freq': 200,
        'max_freq': 4000,
    },
    'input_layernorm': True
}

# space f0 values equally in parameter space (exponentially in frequency space)
F0_INIT_METHOD = 'equal'