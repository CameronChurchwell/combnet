# Runs all experiments

# Args
# $1 - index of GPU to use

set -e

# Download datasets
python -m combnet.data.download

# Create synthetic datasets
python -m combnet.data.synthesize

# Augment
python -m combnet.data.augment --datasets giantsteps_mtg

# Partition
python -m combnet.partition

# Preprocess
python -m combnet.data.preprocess --dataset giantsteps giantsteps_mtg --gpu $1

# Train and evaluate
python -m combnet.train --dataset giantsteps_mtg --config config/giantsteps-comb.py --gpu $1
python -m combnet.train --dataset giantsteps_mtg --config config/giantsteps-stft-chroma.py --gpu $1
python -m combnet.train --dataset giantsteps_mtg --config config/giantsteps-stft-madmom.py --gpu $1
# python -m combnet.train --dataset giantsteps_mtg --config config/giantsteps-dctconv-madmom-static.py --gpu $1 # just for verification
python -m combnet.train --dataset giantsteps_mtg --config config/giantsteps-dctconv-madmom-learned-filters.py --gpu $1
python -m combnet.train --dataset giantsteps_mtg --config config/giantsteps-dctconv-madmom-learned.py --gpu $1

python -m combnet.train --dataset timit --config config/timit-comb-frame.py --gpu $1
python -m combnet.train --dataset timit --config config/timit-conv-frame.py --gpu $1
python -m combnet.train --dataset timit --config config/timit-sinc-frame.py --gpu $1

# # These perform a search over model sizes as discussed in the paper
while python -m combnet.train --dataset notes --config config/notes-comb-search.py --gpu $1; do :; done
while python -m combnet.train --dataset notes --config config/notes-conv-search.py --gpu $1; do :; done
