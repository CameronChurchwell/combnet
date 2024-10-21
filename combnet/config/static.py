"""Config parameters whose values depend on other config parameters"""
import combnet


###############################################################################
# Directories
###############################################################################


# Location to save dataset partitions
PARTITION_DIR = combnet.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = combnet.ASSETS_DIR / 'checkpoints'

# Default configuration file
DEFAULT_CONFIGURATION = combnet.ASSETS_DIR / 'configs' / 'combnet.py'
