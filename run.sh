# Runs all experiments

# Args
# $1 - index of GPU to use

# Download datasets
python -m NAME.data.download

# Setup experiments
python -m NAME.data.preprocess
python -m NAME.partition

# Train and evaluate
accelerate launch -m NAME.train --config config/config.py
