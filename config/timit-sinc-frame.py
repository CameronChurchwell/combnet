from pathlib import Path
import yapecs
globals().update({k:v for k,v in vars(yapecs.import_from_path('timit-frame', Path(__file__).parent / 'timit-frame.py')).items() if not k.startswith('__')})

MODULE = 'combnet'

from pathlib import Path
CONFIG = Path(__file__).stem

MODEL_CLASS = 'SincNet'

# from functools import partial
# OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.001, momentum=0.9)
PARAM_GROUPS = {
    # 'main': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 1e-4},
    # 'main': {'lr': 1e-3},
    # 'main': {'lr': 0.001},
    'main': {'lr': 0.001},
}

# MODEL_KWARGS = {'n_filters': 16}
# MODEL_KWARGS = {
#     'n_filters': int(Path(__file__).stem.split('-')[-1]),
#     # 'n_filters': 24,
#     # 'n_filters': 32,
#     'comb_kwargs': {
#         # 'min_freq': 20,
#         # 'max_freq': 500,
#         'min_freq': 200,
#         'max_freq': 500,
#     }}

# space f0 values equally in parameter space (exponentially in frequency space)
# F0_INIT_METHOD = 'equal'

# MEMORY_CACHING = True # TODO: remove

RANDOM_SEED = 24