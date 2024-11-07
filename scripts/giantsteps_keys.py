import combnet
from collections import defaultdict

for dataset in ['giantsteps', 'giantsteps_mtg']:

    key_dir = combnet.DATA_DIR / dataset

    key_files = key_dir.glob('*.key')

    key_counts = defaultdict(lambda: 0)

    for file in key_files:
        with open(file, 'r') as f:
            key = f.read()
        key_counts[key] += 1

    print(key_counts)
    print(key_counts.keys())

