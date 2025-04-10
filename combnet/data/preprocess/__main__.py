import yapecs

import combnet


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser()
    parser.add_argument(
        '--datasets',
        default=combnet.DATASETS,
        nargs='+',
        help='The names of the datasets to preprocess')
    parser.add_argument(
        '--gpu',
        default=None,
        help='GPU to use for preprocessing'
    )
    return parser.parse_args()

combnet.data.preprocess.datasets(**vars(parse_args()))
