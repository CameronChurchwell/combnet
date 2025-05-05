import json
import os
import random

from pathlib import Path
import re

import combnet

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import penn
else:
    from combnet import penn

def datasets(datasets=combnet.DATASETS, exclude_pattern=None):
    """Partition datasets"""
    for dataset in datasets:

        # Handle penn datasets
        if dataset in ['ptdb', 'mdb']:
            penn.partition.dataset(dataset)
            continue

        # Clear lengths cache
        lengths_files = (combnet.CACHE_DIR / dataset).glob('*-lengths.json')
        for lf in lengths_files:
            os.unlink(lf)

        # Use maestro default split
        if dataset == 'maestro': continue


        # Random seed
        random.seed(combnet.RANDOM_SEED)

        dataset_dir: Path = combnet.DATA_DIR / dataset

        audio_files = dataset_dir.glob('*.wav')

        if exclude_pattern is not None:
            regex = re.compile(exclude_pattern)
            audio_files = [
                f for f in audio_files if not regex.search(str(f))
            ]

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
