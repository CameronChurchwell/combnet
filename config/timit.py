# This is a partial config file that is imported by other config files and should not be used directly.

SAMPLE_RATE = 16_000
N_FFT = 1024
WINDOW_SIZE = 1_600
HOPSIZE = 800
# SAMPLE_RATE = 16_000

EVALUATION_DATASETS = ['timit']

# METRICS = ['accuracy', 'loss', 'categorical']
METRICS = ['accuracy', 'loss']

FEATURES = ['audio', 'class']

# CHORDS = ["C", "F", "Bb", "Eb"]

import yapecs

@yapecs.ComputedProperty(True)
def CLASS_MAP():
    import combnet
    import json
    speakers_file = combnet.DATA_DIR / 'timit' / 'speakers.json'
    assert speakers_file.exists()
    with open (speakers_file, 'r') as f:
        SPEAKERS = json.load(f)
    return {k: i for i, k in enumerate(SPEAKERS)}

@yapecs.ComputedProperty(True)
def NUM_CLASSES():
    import combnet
    print('Number of speakers:', len(combnet.CLASS_MAP))
    return len(combnet.CLASS_MAP)

MODEL_MODULE = 'speaker_classifiers'

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 1_000  # steps

# Number of steps between evaluation tensorboard logging
EVALUATION_INTERVAL = 100  # steps

# Number of training steps
STEPS = 20_000

BATCH_SIZE = 8