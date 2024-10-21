import torchutil

import NAME


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self):
        self.loss = Loss()

    def __call__(self):
        return self.loss()

    def update(self, logits, target):
        # Detach from graph
        logits = logits.detach()

        # Update loss
        self.loss.update(logits, target)

    def reset(self):
        self.loss.reset()


###############################################################################
# Individual metric
###############################################################################


class Loss(torchutil.metrics.Average):
    """Batch-updating loss"""
    def update(self, predicted, target):
        super().update(NAME.loss(predicted, target), target.numel())
