import torchutil

import combnet


###############################################################################
# Aggregate metric
###############################################################################

class Metrics():
    def __init__(self):
        self.accuracy = torchutil.metrics.Accuracy()
        self.loss = Loss()
        self.metrics = [
            self.accuracy,
            self.loss
        ]
        self.reset()

    def __call__(self):
        results = {
            'loss': self.loss(),
            'accuracy': self.accuracy()
        }
        return results

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(
        self, predicted_logits, target_indices):
        self.accuracy.update(predicted_logits.argmax(1), target_indices)
        self.loss.update(predicted_logits, target_indices)


###############################################################################
# Individual metric
###############################################################################


class Loss(torchutil.metrics.Average):
    """Batch-updating loss"""
    def update(self, predicted, target):
        super().update(combnet.loss(predicted, target), target.numel())
