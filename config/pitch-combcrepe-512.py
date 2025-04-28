from pathlib import Path

MODULE = 'combnet'

n_filters = int(Path(__file__).stem.split('-')[-1])

CONFIG = Path(__file__).stem

MODEL_MODULE = 'pitch_estimation'

MODEL_CLASS = 'CombCrepe'

MODEL_KWARGS = {
    'n_filters': n_filters
}

F0_INIT_METHOD = 'equal'