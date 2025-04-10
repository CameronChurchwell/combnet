import yapecs
from pathlib import Path

MODULE = 'combnet'

# Import module, but delay updating search params until after configuration
import combnet
if hasattr(combnet, 'defaults'):

    # Lock file to track search progress
    progress_file = Path(__file__).parent / 'giantsteps-conv-multirun.progress'

    # Values that we want to search over
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    features = ['madmom']
    run = list(range(0, 10))

    # Get grid search parameters for this run
    MODEL_KWARGS_FEATURES, RANDOM_SEED = yapecs.grid_search(
        progress_file,
        features,
        run)

    CONFIG = f'giantsteps-conv-{MODEL_KWARGS_FEATURES}-{RANDOM_SEED}'

    SAMPLE_RATE = 44_100

    HOPSIZE = (SAMPLE_RATE // 5)

    N_FFT = 8192

    FEATURES = ['spectrogram', 'class']

    GIANTSTEPS_KEYS = ['E minor','F minor', 'G minor', 'Db minor', 'C minor', 'Ab major', 'Eb minor', 'G major', 'Bb minor', 'A minor', 'C major', 'D minor', 'Ab minor', 'F major', 'Gb minor', 'B minor', 'Eb major', 'Bb major', 'A major', 'B major', 'D major', 'E major', 'Gb major', 'Db major']

    CLASS_MAP = {k: i for i, k in enumerate(GIANTSTEPS_KEYS)}

    NUM_CLASSES = len(GIANTSTEPS_KEYS)

    MODEL_MODULE = 'key_classifiers'

    MODEL_CLASS = 'STFTClassifier'

    STEPS = 100_000

    EVALUATION_INTERVAL = 1250

    import torch
    from functools import partial
    OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

    MODEL_KWARGS = {'features': MODEL_KWARGS_FEATURES}