import argparse
import shutil
from pathlib import Path

import combnet


###############################################################################
# Entry point
###############################################################################


def main(config, dataset, gpu=None):
    """Train from configuration"""
    # Create output directory
    directory = combnet.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    combnet.train(dataset, directory, gpu=gpu)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config',
        type=Path,
        default=combnet.DEFAULT_CONFIGURATION,
        help='The configuration file')
    parser.add_argument(
        '--dataset',
        default='giantsteps',
        help='The datasets to train on')
    parser.add_argument(
        '--gpu',
        default=None,
        help='The gpu index to use')

    return parser.parse_args()


main(**vars(parse_args()))
