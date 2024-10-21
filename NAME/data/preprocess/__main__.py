import yapecs

import NAME


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser()
    parser.add_argument(
        '--datasets',
        default=NAME.DATASETS,
        nargs='+',
        help='The names of the datasets to preprocess')
    return parser.parse_args()


NAME.preprocess.datasets(**vars(parse_args()))
