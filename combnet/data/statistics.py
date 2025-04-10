from collections import defaultdict
import yapecs

import combnet

###############################################################################
# Download datasets
###############################################################################


def statistics(dataset, partition=None):
    ds = combnet.data.Dataset(dataset, partition, features=combnet.FEATURES)
    print(f"Dataset '{dataset}' (partition '{partition}') statistics:")
    print(f"Length: {len(ds)}")
    class_counts = defaultdict(lambda: 0)
    inverse_map = {i:k for k, i in combnet.CLASS_MAP.items()}
    if 'class' in combnet.FEATURES:
        class_index = combnet.FEATURES.index('class')
        for row in ds:
            c_idx = row[class_index].item()
            c = inverse_map[c_idx]
            class_counts[c] += 1
    print(f'Class counts: {dict(class_counts)}')


if __name__ == '__main__':
    def parse_args():
        """Parse command-line arguments"""
        parser = yapecs.ArgumentParser(description='Synthesize datasets')
        parser.add_argument(
            '--dataset',
            default=combnet.ALL_DATASETS,
            help='The datasets to compute statistics for')
        parser.add_argument(
            '--partition',
            default=None,
            help='The partition to compute statistics for'
        )
        return parser.parse_args()

    statistics(**vars(parse_args()))