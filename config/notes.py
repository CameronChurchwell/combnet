# This is a partial config file that is imported by other config files and should not be used directly.

SAMPLE_RATE = 44_100

EVALUATION_DATASETS = ['notes']

HOPSIZE = SAMPLE_RATE // 10

WINDOW_SIZE = SAMPLE_RATE // 5
# N_FFT = 8192
N_FFT = WINDOW_SIZE

FEATURES = ['audio', 'labels']

MODEL_MODULE = 'piano_transcription'

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 5000  # steps

BATCH_SIZE = 16

# Number of steps between evaluation tensorboard logging
EVALUATION_INTERVAL = 250  # steps

# Number of training steps
STEPS = 10_000

METRICS = ['perclass', 'loss', 'hamming']

import torch
OPTIMIZER_FACTORY = torch.optim.Adam

bce = torch.nn.BCEWithLogitsLoss(reduction='none')
def LOSS_FUNCTION(logits, targets):
    import combnet
    loss = bce(logits, targets)
    mask = targets!=combnet.MASK_INDEX
    loss *= mask
    return loss.sum() / mask.sum()