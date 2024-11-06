MODULE = 'combnet'

CONFIG = 'giantsteps-comb'

SAMPLE_RATE = 44_100

HOP_LENGTH = (SAMPLE_RATE // 5)

N_FFT = 8192

FEATURES = ['audio', 'class']

GIANTSTEPS_KEYS = ['E minor','F minor', 'G minor', 'Db minor', 'C minor', 'Ab major', 'Eb minor', 'G major', 'Bb minor', 'A minor', 'C major', 'D minor', 'Ab minor', 'F major', 'Gb minor', 'B minor', 'Eb major', 'Bb major', 'A major', 'B major', 'D major', 'E major', 'Gb major', 'Db major']

CLASS_MAP = {k: i for i, k in enumerate(GIANTSTEPS_KEYS)}

NUM_CLASSES = len(GIANTSTEPS_KEYS)

NUM_COMB_FILTERS = 24

MODEL_MODULE = 'classifiers'

MODEL_CLASS = 'CombClassifier'

MODEL_KWARGS = {'n_classes': NUM_CLASSES, 'n_filters': NUM_COMB_FILTERS}
