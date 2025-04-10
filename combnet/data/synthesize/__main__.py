import yapecs

import combnet


###############################################################################
# Download datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Synthesize datasets')
    parser.add_argument(
        '--datasets',
        default=combnet.SYNTHETIC_DATASETS,
        nargs='+',
        help='The datasets to synthesize')
    return parser.parse_args()


combnet.data.synthesize.datasets(**vars(parse_args()))
