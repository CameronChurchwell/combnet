###############################################################################
# Configuration
###############################################################################


# Default configuration parameters to be modified
from .config import defaults

# Modify configuration
import yapecs
yapecs.configure('combnet', defaults)
del defaults

# Import configuration parameters
from .config.defaults import *
from .config.static import *


###############################################################################
# Module imports
###############################################################################

from .core import *

from . import _C
from . import filters
from . import functional
from . import modules

from . import models

from .model import Model
# from .train import loss, train
from .train import train, loss
from . import data
from . import evaluate
from . import load
from . import partition
from . import plot
