import json

import NAME


###############################################################################
# Loading utilities
###############################################################################


def partition(dataset):
    """Load partitions for dataset"""
    with open(NAME.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
