"""Config parameters whose values depend on other config parameters"""
import NAME


###############################################################################
# Directories
###############################################################################


# Location to save dataset partitions
PARTITION_DIR = NAME.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = NAME.ASSETS_DIR / 'checkpoints'

# Default configuration file
DEFAULT_CONFIGURATION = NAME.ASSETS_DIR / 'configs' / 'NAME.py'
