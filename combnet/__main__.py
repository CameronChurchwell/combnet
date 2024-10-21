import yapecs
from pathlib import Path

import combnet


###############################################################################
# Command-line interface
###############################################################################


def parse_args():
    parser = yapecs.ArgumentParser()
    parser.add_argument(
        '--input_files',
        nargs='+',
        required=True,
        type=Path,
        help='Input files to process')
    parser.add_argument(
        '--output_files',
        nargs='+',
        required=True,
        type=Path,
        help='Corresponding files to save processed inputs')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=combnet.DEFAULT_CHECKPOINT,
        help='The model checkpoint')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The GPU index')
    return parser.parse_args()


if __name__ == '__main__':
    combnet.from_files_to_files(**vars(parse_args()))
