import torch
import torchutil

from typing import List

import combnet

# Adapted from promonet repo, original code written by Max Morrison

MAX_SHIFTS = (-4, +7)

###############################################################################
# Data augmentation
###############################################################################

@torchutil.notify('augment')
def datasets(datasets: List[str]):
    """Perform data augmentation on cached datasets"""
    for dataset in datasets:

        # Get data directory
        directory = combnet.DATA_DIR / dataset

        # Get files
        audio_files = sorted(directory.glob('*.wav'))
        audio_files = [f for f in audio_files if ('-' not in f.name and '+' not in f.name)]

        # Augment
        from_files_to_files(audio_files)


def from_files_to_files(audio_files):
    """Perform data augmentation on audio files"""
    torch.manual_seed(combnet.RANDOM_SEED)

    shifts = list(range(MAX_SHIFTS[0], 0)) + list(range(1, MAX_SHIFTS[1]+1))

    file_shift_pairs = sum([
        [(audio_file, shift) for shift in shifts] for audio_file in audio_files
    ], [])

    audio_files, shifts = zip(*file_shift_pairs)
    key_files = [file.parent / (file.stem + '.key') for file in audio_files]

    # Get locations to save output
    output_files = [
        file.parent /
        f'{file.stem}{"+" if shift > 0 else ""}{shift}.wav'
        for file, shift in file_shift_pairs]

    output_key_files = [
        file.parent / (file.stem + '.key') for file in output_files
    ]

    # Augment
    combnet.data.augment.pitch.from_files_to_files(
        audio_files,
        key_files,
        output_files,
        output_key_files,
        shifts)