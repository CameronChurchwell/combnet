# This is a partial config file that is imported by other config files and should not be used directly.

SAMPLE_RATE = 44_100
N_FFT = 8192
# SAMPLE_RATE = 16_000

EVALUATION_DATASETS = ['chords']

METRICS = ['accuracy', 'loss', 'categorical']

HOPSIZE = (SAMPLE_RATE // 5)

FEATURES = ['audio', 'class']

CHORDS = ["C", "F", "Bb", "Eb"]

CLASS_MAP = {k: i for i, k in enumerate(CHORDS)}

NUM_CLASSES = len(CHORDS)

MODEL_MODULE = 'chord_classifiers'

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 1_000  # steps

# Number of steps between evaluation tensorboard logging
EVALUATION_INTERVAL = 50  # steps

# Number of training steps
STEPS = 10_000