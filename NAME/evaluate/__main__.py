import yapecs
from pathlib import Path

import NAME


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser()
    parser.add_argument(
        '--datasets',
        default=NAME.EVALUATION_DATASETS,
        nargs='+',
        help='The datasets to evaluate')
    parser.add_argument(
        '--checkpoint',
        default=NAME.DEFAULT_CHECKPOINT,
        type=Path,
        help='The checkpoint file to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')

    return parser.parse_args()


NAME.evaluate.datasets(**vars(parse_args()))
