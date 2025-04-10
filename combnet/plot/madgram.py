import torch
import combnet
from matplotlib import pyplot as plt
from matplotlib import colors

from madmom.audio.filters import LogarithmicFilterbank
import numpy as np

def madmom_filters(bands_per_octave=24):
    return LogarithmicFilterbank(
        np.linspace(0, combnet.SAMPLE_RATE // 2, combnet.N_FFT//2+1),
        num_bands=bands_per_octave,
        fmin=65,
        fmax=2100,
        unique_filters=True
    )

def madgram(audio, norm=True):
    filters = madmom_filters()
    filters_tensor = torch.tensor(filters).T
    # plt.pcolormesh(filters_tensor[:, :200], norm=colors.PowerNorm(0.2)); plt.show()
    with torch.no_grad():
        output = combnet.data.preprocess.spectrogram.from_audio(
            audio=audio,
            sample_rate=combnet.SAMPLE_RATE
        ).cpu().squeeze()
        output = filters_tensor @ output
    output = output[:68]
    output = torch.log(1+output)
    output = output/abs(output).max()
    plt.pcolormesh(
        np.linspace(0, audio.shape[-1] / combnet.SAMPLE_RATE, output.shape[1]),
        filters.center_frequencies[:68],
        output,
        norm=colors.PowerNorm(0.4) if norm else None
    )
    plt.title('MadGram')
    plt.ylabel('Frequencies (Hz)')
    plt.xlabel('Time (Seconds)')
    plt.colorbar()
