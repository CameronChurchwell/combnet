import json

import torch
import torchutil

import combnet


###############################################################################
# Evaluate
###############################################################################


@torchutil.notify('evaluate')
def datasets(
    datasets=combnet.EVALUATION_DATASETS,
    checkpoint=combnet.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Containers for results
    overall, granular = {}, {}

    # Per-file metrics
    file_metrics = combnet.evaluate.Metrics()

    # Per-dataset metrics
    dataset_metrics = combnet.evaluate.Metrics()

    # Aggregate metrics over all datasets
    aggregate_metrics = combnet.evaluate.Metrics()

    # Evaluate each dataset
    for dataset in datasets:

        # Reset dataset metrics
        dataset_metrics.reset()

        # Iterate over test set
        for batch in torchutil.iterator(
            combnet.data.loader(dataset, 'test'),
            f'Evaluating {combnet.CONFIG} on {dataset}'
        ):

            # Reset file metrics
            file_metrics.reset()

            # TODO - unpack
            () = batch

            # TODO - copy to device

            # TODO - inference

            # Update metrics
            args = (
                # TODO - args
            )
            file_metrics.update(*args)
            dataset_metrics.update(*args)
            aggregate_metrics.update(*args)

            # Save results
            granular[f'{dataset}/{stem[0]}'] = file_metrics()
        overall[dataset] = dataset_metrics()
    overall['aggregate'] = aggregate_metrics()

    # Write to json files
    directory = combnet.EVAL_DIR / combnet.CONFIG
    directory.mkdir(exist_ok=True, parents=True)
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)
