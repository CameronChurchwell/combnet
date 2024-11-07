import json
import random

from pathlib import Path

import combnet


def datasets(datasets=combnet.DATASETS):
    """Partition datasets"""
    for dataset in datasets:

        # Random seed
        random.seed(combnet.RANDOM_SEED)

        dataset_dir: Path = combnet.DATA_DIR / dataset

        audio_files = dataset_dir.glob('*.wav')

        stems = [f.stem for f in audio_files]

        random.shuffle(stems)

        num_examples = len(stems)

        if dataset == 'giantsteps_mtg':
            cutoffs = int(num_examples * 0.9), int(num_examples * 1) # 90-10 train-valid
        elif dataset == 'giantsteps':
            cutoffs = int(num_examples * 0), int(num_examples * 0) # testing only
        else:
            cutoffs = int(num_examples * 0.8), int(num_examples * 0.9)

        partition = {
            'train': stems[:cutoffs[0]],
            'valid': stems[cutoffs[0]:cutoffs[1]],
            'test': stems[cutoffs[1]:]
        }

        # Save to disk
        file = combnet.PARTITION_DIR / f'{dataset}.json'
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, 'w') as file:
            json.dump(partition, file, indent=4)
