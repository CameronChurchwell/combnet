import contextlib
import multiprocessing as mp
import time

from pathlib import Path

import torch
import torchutil

import combnet

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import penn
else:
    from combnet import penn

###############################################################################
# Preprocess
###############################################################################


@torchutil.notify('preprocess')
def datasets(
    datasets=combnet.DATASETS,
    features=None,
    gpu=None,
    num_workers=0,
    partition=None):
    """Preprocess a dataset

    Arguments
        datasets
            The names of the dataset to preprocess
        features
            The names of the features preprocess
        gpu
            The gpu to use for preprocessing
        num_workers
            The number of worker threads to use
        partition
            The partition to preprocess. Default (None) uses all partitions.
    """
    if features is None:
        features = combnet.FEATURES

    for dataset in datasets:

        if dataset in ['ptdb', 'mdb']:
            penn.data.preprocess.datasets([dataset])
            continue

        try:

            # Setup multiprocessed dataloader
            dataloader = combnet.data.loader(
                dataset,
                partition,
                features=['audio', 'length', 'audio_file'],
                num_workers=num_workers // 2)

        except ValueError:

            # Empty partition
            continue

        output = {
            file: f'{Path(str(file.parent).replace("/datasets/", "/cache/"))}/{file.stem}' + '-{}.pt'
            for _, _, files in dataloader for file in files}
        from_dataloader(
            dataloader,
            features,
            output,
            num_workers=(num_workers + 1) // 2,
            gpu=gpu)


def from_files_to_files(
    audio_files,
    output_files,
    features=combnet.FEATURES,
    num_workers=0,
    gpu=None):
    """Preprocess from files

    Arguments
        audio_files
            A list of audio files to process
        output_files
            A list of output files to use to save features
        features
            The names of the features to do preprocessing for
        num_workers
            The number of worker threads to use
        gpu
            The gpu to use for preprocessing
    """
    # Setup dataloader
    dataloader = combnet.data.loader(
        audio_files,
        features=['audio', 'length', 'audio_file'],
        num_workers=num_workers//2)
    from_dataloader(
        dataloader,
        features,
        dict(zip(audio_files, output_files)),
        num_workers=(num_workers+1)//2,
        gpu=gpu)


###############################################################################
# Utilities
###############################################################################


def from_dataloader(loader, features, output, num_workers=0, gpu=None):
    """Preprocess from a dataloader

    Arguments
        loader
            A Pytorch DataLoader yielding batches of (audio, length, filename)
        features
            The names of the features to preprocess
        output
            A dictionary mapping audio filenames to output filenames
        num_workers
            The number of worker threads to use for async file saving
        gpu
            The gpu to use for preprocessing
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Setup multiprocessing
    if num_workers == 0:
        pool = contextlib.nullcontext()
    else:
        pool = mp.get_context('spawn').Pool(num_workers)

    try:

        with torch.inference_mode():

            # Batch preprocess
            for audios, lengths, audio_files in torchutil.iterator(
                loader,
                f'Preprocessing {", ".join(features)} '
                f'for {loader.dataset.metadata.name}',
                total=len(loader)
            ):
                # Copy to device
                audios = audios.to(device)
                lengths = lengths.to(device)

                for feature in features:

                    # Preprocess
                    outputs = getattr(
                        combnet.data.preprocess,
                        feature
                    ).from_audios(audios, lengths, gpu=gpu).cpu()

                    # Get length in frames
                    frame_lengths = lengths // combnet.HOPSIZE

                    # Get output filenames
                    filenames = []
                    for file in audio_files:
                        output_file = output[file]
                        if '{}' in output_file:
                            filenames.append(
                                output_file.format(feature))
                        else:
                            filenames.append(output_file)

                    if num_workers == 0:

                        # Synchronous save
                        for latent_output, filename, new_length in zip(
                            outputs.cpu(),
                            filenames,
                            frame_lengths.cpu()
                        ):
                            save_masked(latent_output, filename, new_length)
                    else:

                        # Asynchronous save
                        pool.starmap_async(
                            save_masked,
                            zip(outputs, filenames, frame_lengths.cpu()))

                        # Wait if the queue is full
                        while pool._taskqueue.qsize() > 256:
                            time.sleep(1)

    finally:

        # Shutdown multiprocessing
        if num_workers > 0:
            pool.close()
            pool.join()


def from_audio(
    audio,
    feature=combnet,
    sample_rate=combnet.SAMPLE_RATE,
    gpu=None
):
    """Preprocess audio"""
    audio = combnet.resample(audio, sample_rate)

    if feature is None:
        feature = combnet.feature

    # Compute feature
    with torch.autocast('cuda' if gpu is not None else 'cpu'):
        features = getattr(combnet.preprocess, feature).from_audio(
            audio,
            sample_rate=combnet.SAMPLE_RATE,
            gpu=gpu)

        if features.dim() == 2:
            features = features[None]

        return features


def save_masked(tensor, file, length):
    """Save masked tensor"""
    torch.save(tensor[..., :length].clone(), file)