# This is a partial config file that is imported by other config files and should not be used directly.

SAMPLE_RATE = 44_100
N_FFT = 8192

EVALUATION_DATASETS = ['maestro']

HOPSIZE = (SAMPLE_RATE // 5)

WINDOW_SIZE = 8192

FEATURES = ['audio', 'labels']

MODEL_MODULE = 'piano_transcription'

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 2_500  # steps

BATCH_SIZE = 4

# Number of steps between evaluation tensorboard logging
EVALUATION_INTERVAL = 500  # steps

# Number of training steps
STEPS = 20_000

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