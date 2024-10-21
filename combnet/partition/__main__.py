import yapecs

import combnet


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        default=combnet.DATASETS,
        nargs='+',
        help='The datasets to partition')
    return parser.parse_args()


combnet.partition.datasets(**vars(parse_args()))
