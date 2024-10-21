import torchutil

import combnet


###############################################################################
# Preprocess
###############################################################################


@torchutil.notify('preprocess')
def datasets(datasets):
    """Preprocess a dataset

    Arguments
        name - string
            The name of the dataset to preprocess
    """
    for dataset in datasets:
        input_directory = combnet.DATA_DIR / dataset
        output_directory = combnet.CACHE_DIR / dataset

        # TODO - Perform preprocessing
        raise NotImplementedError
