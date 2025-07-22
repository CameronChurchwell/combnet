import contextlib
from typing import Union

import torch

import torchaudio

import combnet


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def inference_context(model):
    # device_type = next(model.parameters()).device.type

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.inference_mode():
        yield

    # Prepare model for training
    model.train()


def resample(
    audio: torch.Tensor,
    sample_rate: Union[int, float],
    target_rate: Union[int, float] = combnet.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)