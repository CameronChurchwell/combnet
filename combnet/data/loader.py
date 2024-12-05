import torch

import combnet


def loader(
    dataset, 
    partition, 
    features=combnet.FEATURES, 
    num_workers=0, #TODO make this override?
    batch_size=None,
    gpu=None):
    """Retrieve a data loader"""
    dataset=combnet.data.Dataset(
        name_or_files=dataset,
        partition=partition, 
        features=features
    )
    collate = combnet.data.Collate(features=features)
    test = 'test' in partition
    if test:
        batch_size = 1
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=combnet.BATCH_SIZE if batch_size is None else batch_size,
        shuffle='train' in partition or 'valid' in partition,
        num_workers=combnet.NUM_WORKERS,
        pin_memory=(gpu is not None),
        collate_fn=collate)
