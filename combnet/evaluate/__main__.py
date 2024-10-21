import yapecs
from pathlib import Path

import combnet


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser()
    parser.add_argument(
        '--datasets',
        default=combnet.EVALUATION_DATASETS,
        nargs='+',
        help='The datasets to evaluate')
    parser.add_argument(
        '--checkpoint',
        default=combnet.DEFAULT_CHECKPOINT,
        type=Path,
        help='The checkpoint file to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')

    return parser.parse_args()


combnet.evaluate.datasets(**vars(parse_args()))
