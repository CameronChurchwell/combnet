import torch

import NAME


def loader(datasets, partition, gpu=None):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=NAME.data.Dataset(datasets, partition),
        batch_size=NAME.BATCH_SIZE,
        shuffle=partition == 'train',
        num_workers=NAME.NUM_WORKERS,
        pin_memory=gpu is not None,
        collate_fn=NAME.data.collate)
