MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

MODEL_MODULE = 'pitch_estimation'

MODEL_CLASS = 'CombFcnf0'

F0_INIT_METHOD = 'equal'

MODEL_KWARGS = {
    'n_filters': 32
}