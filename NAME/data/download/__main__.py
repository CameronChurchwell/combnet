import yapecs

import NAME


###############################################################################
# Download datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Download datasets')
    parser.add_argument(
        '--datasets',
        default=NAME.DATASETS,
        nargs='+',
        help='The datasets to download')
    return parser.parse_args()


NAME.data.download.datasets(**vars(parse_args()))
