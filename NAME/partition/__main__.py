import yapecs

import NAME


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        default=NAME.DATASETS,
        nargs='+',
        help='The datasets to partition')
    return parser.parse_args()


NAME.partition.datasets(**vars(parse_args()))
