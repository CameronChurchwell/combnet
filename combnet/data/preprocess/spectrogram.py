import multiprocessing as mp
import os

import torch

import combnet


###############################################################################
# Spectrogram computation
###############################################################################


def from_audios(audio, lengths, gpu=None):
    if (
        not hasattr(from_audio, 'window') or
        from_audio.dtype != audio.dtype or
        from_audio.device != audio.device
    ):
        from_audio.window = torch.hann_window(
            combnet.WINDOW_SIZE,
            dtype=audio.dtype,
            device=audio.device)
        from_audio.dtype = audio.dtype
        from_audio.device = audio.device

    # Pad audio
    size = (combnet.N_FFT - combnet.HOPSIZE) // 2
    audio = torch.nn.functional.pad(
        audio,
        (size, size),
        mode='reflect')

    # Compute stft
    stft = torch.stft(
        audio.squeeze(1),
        n_fft=combnet.N_FFT,
        hop_length=combnet.HOPSIZE,
        win_length=combnet.WINDOW_SIZE,
        window=from_audio.window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True)
    spectrogram = abs(stft)

    return spectrogram#.to(torch.float16)


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
    torch.save(output, output_file)


def from_files_to_files(audio_files, output_files):
    """Compute spectrogram from audio files and save to disk"""
    with mp.get_context('spawn').Pool(os.cpu_count() // 2) as pool:
        pool.starmap(from_file_to_file, zip(audio_files, output_files))