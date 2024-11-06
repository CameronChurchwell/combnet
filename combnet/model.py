import torch

import combnet

###############################################################################
# Model
###############################################################################

def Model():
    """Create model based on config"""

    try:
        module = getattr(combnet.models, combnet.MODEL_MODULE)
    except:
        # TODO improve these
        raise ValueError(f'Could not find model module "{combnet.MODEL_MODULE}"')

    try:
        model_class = getattr(module, combnet.MODEL_CLASS)
    except:
        raise ValueError(f'Could not load model class {combnet.MODEL_CLASS} from module {combnet.MODEL_MODULE}')
    
    return model_class(**combnet.MODEL_KWARGS)
