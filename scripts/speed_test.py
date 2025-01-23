import combnet
import time
import torch
import torchaudio
import argparse
from tqdm import trange

def speedtest(implementation, gpu=None, inference=False, backward=False, big=False):
    if inference and backward:
        raise ValueError('Cannot have both inference=True and backward=True')

    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    audio_path = combnet.DATA_DIR / 'giantsteps_mtg' / '1093726.LOFI.wav'

    audio, sr = torchaudio.load(audio_path)

    audio = audio.mean(0, keepdim=True)

    audio = audio.to(device)

    if big:
        audio = audio[None].repeat(4, 2, 1)
        f0 = torch.tensor([[151.1, 350.7]]).to(device)
        a = torch.tensor(0.8).to(device)
    else:
        f0 = torch.tensor(151.1).to(device)
        a = torch.tensor(0.8).to(device)
    sr = torch.tensor(sr).to(device)

    print(audio.shape, sr)

    comb_fn = getattr(combnet.filters, implementation)

    f0.requires_grad_()

    # warmup
    if inference:
        with torch.no_grad():
            _ = comb_fn(audio, f0, a, sr)
    elif backward:
        out = comb_fn(audio, f0, a, sr)
        pseudo_loss = out.sum()
        if f0.grad is not None:
            f0.grad.zero_()
        pseudo_loss.backward()
    else:
        _ = comb_fn(audio, f0, a, sr)

    torch.cuda.synchronize()
    start = time.time()
    max_iters = 1000
    for _ in trange(0, max_iters, desc=f'speed testing {implementation}'):
        if inference:
            with torch.no_grad():
                _ = comb_fn(audio, f0, a, sr)
        elif backward:
            out = comb_fn(audio, f0, a, sr)
            pseudo_loss = out.sum()
            if f0.grad is not None:
                f0.grad.zero_()
            pseudo_loss.backward()
        else:
            _ = comb_fn(audio, f0, a, sr)


    torch.cuda.synchronize()
    end = time.time()
    seconds = end-start

    print(f"seconds elapsed: {seconds}")
    if gpu is not None:
        print(f"max memory allocated: {torch.cuda.max_memory_allocated(device) / 1024 / 1024 / 1024:0.6f} GB")

if __name__ == '__main__':
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
        '--inference',
        action='store_true',
        help='run with no_grad()'
    )
    parser.add_argument(
        '--backward',
        action='store_true',
        help='Compute backwards pass as well'
    )
    parser.add_argument(
        '--big',
        action='store_true',
        help='Use a larger number of channels and batch size'
    )
    speedtest(**vars(parser.parse_args()))