import torchutil
import combnet
from .chords import chords


###############################################################################
# Synthesize datasets
###############################################################################


@torchutil.notify('synthesize')
def datasets(datasets=combnet.DATASETS):
    """Synthesize datasets"""
    if 'chords' in datasets:
        chords()