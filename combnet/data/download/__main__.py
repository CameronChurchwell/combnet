import yapecs

import combnet


###############################################################################
# Download datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Download datasets')
    parser.add_argument(
        '--datasets',
        default=combnet.DATASETS,
        nargs='+',
        help='The datasets to download')
    return parser.parse_args()


combnet.data.download.datasets(**vars(parse_args()))
