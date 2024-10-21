import json

import combnet


###############################################################################
# Loading utilities
###############################################################################


def partition(dataset):
    """Load partitions for dataset"""
    with open(combnet.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
