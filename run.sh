# Runs all experiments

# Args
# $1 - index of GPU to use

# Download datasets
python -m combnet.data.download

# Setup experiments
python -m combnet.data.preprocess
python -m combnet.partition

# Train and evaluate
accelerate launch -m combnet.train --config config/config.py
