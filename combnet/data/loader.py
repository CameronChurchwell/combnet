import torch

import combnet


def loader(datasets, partition, gpu=None):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=combnet.data.Dataset(datasets, partition),
        batch_size=combnet.BATCH_SIZE,
        shuffle=partition == 'train',
        num_workers=combnet.NUM_WORKERS,
        pin_memory=gpu is not None,
        collate_fn=combnet.data.collate)
