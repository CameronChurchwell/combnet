import yapecs

import combnet


###############################################################################
# Data augmentation
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Perform data augmentation')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=combnet.DATASETS,
        help='The name of the datasets to augment')
    return parser.parse_args()


combnet.data.augment.datasets(**vars(parse_args()))