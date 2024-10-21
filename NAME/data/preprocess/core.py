import torchutil

import NAME


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
        input_directory = NAME.DATA_DIR / dataset
        output_directory = NAME.CACHE_DIR / dataset

        # TODO - Perform preprocessing
        raise NotImplementedError
