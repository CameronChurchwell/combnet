import torch

import combnet


def loader(
    dataset, 
    partition, 
    features=combnet.FEATURES, 
    num_workers=None,
    batch_size=None,
    gpu=None):
    """Retrieve a data loader"""
    dataset=combnet.data.Dataset(
        name_or_files=dataset,
        partition=partition, 
        features=features
    )
    collate = combnet.data.Collate(features=features)
    test = 'test' in partition if partition is not None else False
    valid = 'valid' in partition if partition is not None else False
    if test:
        batch_size = 1
    if valid:
        batch_size = 1
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=combnet.BATCH_SIZE if batch_size is None else batch_size,
        shuffle='train' in partition or 'valid' in partition if partition is not None else False,
        num_workers=combnet.NUM_WORKERS if num_workers is None else num_workers,
        pin_memory=(gpu is not None),
        collate_fn=collate
    )
