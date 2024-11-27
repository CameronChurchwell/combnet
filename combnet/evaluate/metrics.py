import torchutil
import torch

import combnet


###############################################################################
# Aggregate metric
###############################################################################

class Metrics():
    def __init__(self):
        self.accuracy = torchutil.metrics.Accuracy()
        self.loss = Loss()
        self.mirex_weighted = MIREX_Weighted()
        self.categorical = CategoricalAccuracy()
        self.metrics = [
            self.accuracy,
            self.mirex_weighted,
            self.loss,
            self.categorical
        ]
        self.reset()

    def __call__(self):
        results = {
            'loss': self.loss(),
            'accuracy': self.accuracy(),
            'mirex_weighted': self.mirex_weighted(),
        }
        results = results | self.categorical()
        return results

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(
        self, predicted_logits, target_indices):
        predicted_indices = predicted_logits.argmax(1)
        self.accuracy.update(predicted_indices, target_indices)
        self.loss.update(predicted_logits, target_indices)
        self.mirex_weighted.update(predicted_indices, target_indices)
        self.categorical.update(predicted_logits, target_indices)


###############################################################################
# Individual metric
###############################################################################


class Loss(torchutil.metrics.Average):
    """Batch-updating loss"""
    def update(self, predicted, target):
        super().update(combnet.loss(predicted, target), target.numel())


class CategoricalAccuracy:

    def __init__(self):
        self.reset()
        self.map = {i: key for i, key in enumerate(combnet.GIANTSTEPS_KEYS)}

    def __call__(self):
        if self.totals is not None:
            assert self.totals.shape == self.counts.shape
        else:
            return None
        output = {}
        for i in range(0, self.totals.shape[0]):
            output[f'categorical_accuracy/{self.map[i]}'] = (
                self.totals[i] / self.counts[i]).item()
            output[f'categorical_total/{self.map[i]}'] = self.totals[i].item()
            output[f'categorical_count/{self.map[i]}'] = self.counts[i].item()
        return output

    def reset(self):
        self.totals = None
        self.counts = None

    def update(self, predicted_logits, target_indices):
        """Update per-category accuracy"""
        # Get predicted category
        predicted_indices = predicted_logits.argmax(dim=1)
        predicted_onehots = torch.nn.functional.one_hot(
            predicted_indices,
            num_classes=predicted_logits.shape[-1])

        # Get target category
        target_onehots = torch.nn.functional.one_hot(
            target_indices,
            num_classes=predicted_logits.shape[-1])

        # Update totals
        marginal_totals = torch.mul(
            predicted_onehots,
            target_onehots).sum(dim=0)
        if self.totals is None:
            self.totals = marginal_totals
        else:
            self.totals += marginal_totals

        # Update counts
        marginal_counts = target_onehots.sum(dim=0)
        if self.counts is None:
            self.counts = marginal_counts
        else:
            self.counts += marginal_counts




SEMITONES = [
    'Ab',
    'A',
    'Bb',
    'B',
    'C',
    'Db',
    'D',
    'Eb',
    'E',
    'F',
    'Gb',
    'G'
]

class MIREX_Weighted(torchutil.metrics.Metric):
    """Batch-updating weighted MIREX accuracy"""

    def update(self, predicted_indices, target_indices):
        predicted_keys = [combnet.GIANTSTEPS_KEYS[predicted_index] for predicted_index in predicted_indices]
        target_keys = [combnet.GIANTSTEPS_KEYS[predicted_index] for predicted_index in target_indices]
        self.total += len(predicted_indices)
        for pred, target in zip(predicted_keys, target_keys):
            pred_tonic, pred_mode = pred.split()
            target_tonic, target_mode = target.split()
            pred_semi_index = SEMITONES.index(pred_tonic)
            target_semi_index = SEMITONES.index(target_tonic)
            if pred == target:
                self.correct += 1
            elif pred_mode == target_mode and (
                (pred_semi_index + 7) % len(SEMITONES) == target_semi_index or\
                (target_semi_index + 7) % len(SEMITONES) == pred_semi_index
            ):
                self.fifth += 1
            elif pred_mode != target_mode and pred_tonic == target_tonic:
                self.parallel += 1
            elif pred_mode != target_mode and (
                (pred_mode == 'minor' and (pred_semi_index + -3) % len(SEMITONES) == target_semi_index) or\
                (pred_mode == 'major' and (target_semi_index + 3) % len(SEMITONES) == pred_semi_index)
            ):
                self.relative += 1

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
