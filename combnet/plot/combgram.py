import torch
import combnet
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

def combgram(audio, comb_fn, sample_rate=None):
    if sample_rate is None:
        sample_rate = combnet.SAMPLE_RATE

    with torch.no_grad():
        output = comb_fn(audio)
        output = torch.nn.functional.max_pool1d(
            output,
            combnet.WINDOW_SIZE,
            combnet.HOPSIZE,
            # combnet.WINDOW_SIZE//2
        ).cpu().squeeze()
    plt.pcolormesh(
        np.linspace(0, audio.shape[-1] / combnet.SAMPLE_RATE, output.shape[1]),
        comb_fn.f.squeeze().cpu().numpy(),
        output,
        # norm=colors.PowerNorm(0.2) # not needed for this?
    )
    plt.title('CombGram')
    plt.ylabel('Frequencies (Hz)')
    plt.xlabel('Time (Seconds)')
    plt.colorbar()

def fused_combgram(audio, fused_comb_fn=None, sample_rate=None, norm=True):
    if sample_rate is None:
        sample_rate = combnet.SAMPLE_RATE
    if fused_comb_fn is None:
        fused_comb_fn = combnet.modules.FusedComb1d()

    with torch.no_grad():
        output = fused_comb_fn(audio).squeeze().cpu()

    output = output/abs(output).max()
    # output = torch.log(1+output)
    # output = output ** 2
    plt.pcolormesh(
        np.linspace(0, audio.shape[-1] / combnet.SAMPLE_RATE, output.shape[1]),
        fused_comb_fn.f.squeeze().detach().cpu().numpy(),
        output,
        norm=colors.PowerNorm(0.2) if norm else None # not needed for this?
    )
    plt.title('(fused) CombGram')
    plt.ylabel('Frequencies (Hz)')
    plt.xlabel('Time (Seconds)')
    plt.colorbar()
