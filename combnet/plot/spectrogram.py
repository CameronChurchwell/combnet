import torch
import combnet
from matplotlib import pyplot as plt
from matplotlib import colors

def spectrogram(audio):
    with torch.no_grad():
        output = combnet.data.preprocess.spectrogram.from_audio(
            audio=audio,
            sample_rate=combnet.SAMPLE_RATE
        ).cpu().squeeze()
    plt.pcolormesh(output, norm=colors.PowerNorm(0.2))
