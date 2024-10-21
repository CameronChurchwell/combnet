import json
import random

import combnet


def datasets(datasets=combnet.DATASETS):
    """Partition datasets"""
    for dataset in datasets:

        # Random seed
        random.seed(combnet.RANDOM_SEED)

        # TODO - make partition dictionary
        partition = {'train': [], 'valid': [], 'test': []}

        # Save to disk
        file = combnet.PARTITION_DIR / f'{dataset}.json'
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, 'w') as file:
            json.dump(partition, file, indent=4)
