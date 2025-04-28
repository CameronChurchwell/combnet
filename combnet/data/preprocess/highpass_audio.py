import multiprocessing as mp
import os

import torch
import torchaudio
import scipy
import scipy.signal

import combnet


###############################################################################
# Spectrogram computation
###############################################################################


def from_audios(audio, lengths, gpu=None):
    if (
        not hasattr(from_audio, 'a') or not hasattr(from_audio, 'b')
    ):
        from_audio.b, from_audio.a = scipy.signal.butter(
            5, # order
            20 / (0.5 * combnet.SAMPLE_RATE), # normalized cutoff
            btype='high',
        )
    device = audio.device
    assert audio.dim() == 3
    audio = audio.squeeze(1).cpu().numpy()
    for i in range(0, audio.shape[0]):
        audio[i] = scipy.signal.lfilter(from_audio.b, from_audio.a, audio[i])

    return torch.tensor(audio).to(device)[:, None, :]


def from_audio(audio, sample_rate, gpu=None):
    if sample_rate != combnet.SAMPLE_RATE:
        audio = combnet.resample(audio, sample_rate, combnet.SAMPLE_RATE)
    if audio.dim() == 2:
        audio = audio.unsqueeze(dim=0)
    return from_audios(audio, audio.shape[-1], gpu=gpu)


def from_file(audio_file):
    """Compute spectrogram from audio file"""
    audio = combnet.load.audio(audio_file)
    return from_audio(audio)


def from_file_to_file(audio_file, output_file):
    """Compute spectrogram from audio file and save to disk"""
    output = from_file(audio_file)
    torchaudio.save(output_file, output)


def from_files_to_files(audio_files, output_files):
    """Compute spectrogram from audio files and save to disk"""
    with mp.get_context('spawn').Pool(os.cpu_count() // 2) as pool:
        pool.starmap(from_file_to_file, zip(audio_files, output_files))