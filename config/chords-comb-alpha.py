from pathlib import Path
import yapecs
import combnet
import os
import torch
import numpy as np

MODULE = 'combnet'

# import chords config
base_config_path = Path(__file__).parent / 'chords.py'
base_config_path = os.getenv(f'{MODULE.upper()}_BASE_CONFIG', base_config_path)
globals().update(vars(yapecs.import_from_path('chords', base_config_path)))

MODEL_CLASS = 'CombClassifier'

RUNS_DIR = Path(__file__).parent.parent / 'runs.alpha'

if hasattr(combnet, 'defaults'):

    def format_float(f):
        return f'{f:.03f}'.replace('.', '_')

    alpha = np.arange(0.8, 1, 0.025).tolist()

    if os.getenv('COMBNET_NO_SEARCH') is None:
        ALPHA, = yapecs.grid_search(
            Path(__file__).parent / 'chords-comb-alpha.progress',
            alpha
        )
    else:
        ALPHA = float(os.getenv('ALPHA'))

    print((ALPHA - 0.8) / 0.2)

    PARAM_GROUPS = {
        'main': {'lr': 5e-5},
        'f0': {'lr': 1e-4}
    }
    OPTIMIZER_FACTORY = torch.optim.Adam

    # MODEL_KWARGS = {'n_filters': 16}
    MODEL_KWARGS = {
        # 'n_filters': 24,
        'n_filters': 32,
        'comb_kwargs': {
            'min_bin': 20,
            'max_bin': 84,
            'min_freq': 25.95,
            'max_freq': 1046.5,
            'alpha': ALPHA
        },
    }

    # Number of steps between evaluation tensorboard logging
    EVALUATION_INTERVAL = 500  # steps

    CONFIG = f'chords-comb-{format_float(ALPHA)}'

    # Number of training steps
    STEPS = 50_000

    F0_INIT_METHOD = 'equal'