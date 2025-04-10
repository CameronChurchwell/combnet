import combnet
import torch
import torchaudio
import argparse
import triton_viz
from triton_viz.interpreter import record_builder
from triton_viz.data import *

def viz(implementation, gpu=None, backward=False):
    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    audio_path = combnet.DATA_DIR / 'giantsteps_mtg' / '1093726.LOFI.wav'

    audio, sr = torchaudio.load(audio_path)

    audio = audio.mean(0, keepdim=True)

    audio = audio.to(device)

    audio = audio[..., :8*3].contiguous().clone()
    audio = audio.reshape(audio.size())

    f0 = torch.tensor(151.1).to(device)
    a = torch.tensor(0.8).to(device)
    sr = torch.tensor(sr).to(device)

    print(audio.shape, sr)

    comb_fn = getattr(combnet.filters, implementation)

    f0.requires_grad_()

    if backward:
        out = comb_fn(audio, f0, a, sr)
        pseudo_loss = out.sum()
        if f0.grad is not None:
            f0.grad.zero_()
        pseudo_loss.backward()
    else:
        _ = comb_fn(audio, f0, a, sr)
    # ops = record_builder.launches[0].records

if __name__ == '__main__':
    # triton_viz.sample((0,))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU to use for testing (default is CPU)'
    )
    parser.add_argument(
        '--implementation',
        type=str,
        default='fractional_comb_fiir',
        help='Which implementation to test'
    )
    parser.add_argument(
        '--backward',
        action='store_true',
        help='Compute backwards pass as well'
    )
    viz(**vars(parser.parse_args()))
    triton_viz.launch()