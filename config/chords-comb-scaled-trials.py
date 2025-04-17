from pathlib import Path
import yapecs
import combnet
import os
import torch

MODULE = 'combnet'

# import chords config
base_config_path = Path(__file__).parent / 'chords.py'
base_config_path = os.getenv(f'{MODULE.upper()}_BASE_CONFIG', base_config_path)
globals().update(vars(yapecs.import_from_path('chords', base_config_path)))

MODEL_CLASS = 'CombClassifier'

RUNS_DIR = Path(__file__).parent.parent / 'runs.chords_comb_trials'

if hasattr(combnet, 'defaults'):

    def format_float(f):
        return str(f).replace('.', '_')

    static = [True, False]
    scaled = [True, False]
    seed = list(range(0, 10))

    if os.getenv('COMBNET_NO_SEARCH') is None:
        STATIC, SCALED, RANDOM_SEED = yapecs.grid_search(
            Path(__file__).parent / 'chords-comb-scaled-trials.progress',
            static,
            scaled,
            seed
        )
    else:
        RANDOM_SEED = float(os.getenv('COMBNET_RANDOM_SEED'))
        STATIC = int(os.getenv('COMBNET_STATIC'))
        SCALED = int(os.getenv('COMBNET_SCALED'))

    PARAM_GROUPS = {
        'main': {'lr': 5e-5},
        'f0': {'lr': 1e-4 if not STATIC else 0}
    }
    OPTIMIZER_FACTORY = torch.optim.Adam

    # MODEL_KWARGS = {'n_filters': 16}
    if SCALED:
        MODEL_KWARGS = {
            'n_filters': 12,
            'comb_kwargs': {
                'min_bin': 20,
                'max_bin': 84,
                'min_freq': 25.95,
                'max_freq': 1046.5
            }
        }
    else:
        MODEL_KWARGS = {'n_filters': 12}

    # Number of steps between evaluation tensorboard logging
    EVALUATION_INTERVAL = 500  # steps

    CONFIG = 'chords-comb'

    if SCALED:
        CONFIG += '-scaled'
    if STATIC:
        CONFIG += '-static'

    CONFIG += f'-{RANDOM_SEED}'

    # Number of training steps
    STEPS = 50_000