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


class MIREX_Weighted(torchutil.metrics.Metric):
    """Batch-updating weighted MIREX accuracy"""

    def update(self, predicted_indices, target_indices):
        pass # TODO fill this in

    def reset(self):
        self.correct = 0
        self.fifth = 0
        self.relative = 0
        self.parallel = 0
        self.total = 0

    def __call__(self):
        r_correct = self.correct/float(self.total)
        r_fifth = self.fifth/float(self.total)
        r_relative = self.relative/float(self.total)
        r_parallel = self.parallel/float(self.total)
        return r_correct + \
               0.5 * r_fifth + \
               0.3 * r_relative + \
               0.2 * r_parallel
