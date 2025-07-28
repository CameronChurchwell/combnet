import torchutil
import torch

import combnet


###############################################################################
# Aggregate metric
###############################################################################

class Metrics():
    def __init__(self):
        # self.accuracy = torchutil.metrics.Accuracy()
        # self.loss = Loss()
        # self.mirex_weighted = MIREX_Weighted()
        # self.categorical = CategoricalAccuracy()
        # self.perclass = PerClassBinaryAccuracy()
        metrics = {
            'accuracy': torchutil.metrics.Accuracy,
            'loss': Loss,
            'mirex_weighted': MIREX_Weighted,
            'categorical':  CategoricalAccuracy,
            'perclass': PerClassBinaryAccuracy,
            'hamming': HammingLoss
        }
        self.metrics = {name:metrics[name]() for name in combnet.METRICS}
        self.reset()

    def __call__(self):
        multi_metrics = ['categorical', 'perclass']
        results = {name: metric() for name, metric in self.metrics.items() if name not in multi_metrics}
        if 'categorical' in self.metrics:
            results = results | self.metrics['categorical']()
        if 'perclass' in self.metrics:
            results = results | self.metrics['perclass']()
        return results

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def update(self, predicted_logits, target_indices):
        for name, metric in self.metrics.items():
            if name in ['accuracy', 'mirex_weighted']:
                predicted_indices = predicted_logits.argmax(1)
                metric.update(predicted_indices, target_indices)
            elif name in ['loss', 'categorical', 'perclass', 'hamming']:
                metric.update(predicted_logits, target_indices)
            else:
                raise ValueError(f'Unable to determine input for metric: {name}')


###############################################################################
# Individual metric
###############################################################################


class Loss(torchutil.metrics.Average):
    """Batch-updating loss"""
    def update(self, predicted, target):
        super().update(combnet.loss(predicted, target).detach(), target.numel())


class CategoricalAccuracy:

    def __init__(self):
        self.reset()
        self.map = {i: key for key, i in combnet.CLASS_MAP.items()}

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
        inverse_map = {i:k for k, i in combnet.CLASS_MAP.items()}
        predicted_keys = [inverse_map[predicted_index.item()] for predicted_index in predicted_indices]
        target_keys = [inverse_map[predicted_index.item()] for predicted_index in target_indices]
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


class PerClassBinaryAccuracy(torchutil.metrics.Metric):

    def reset(self):
        self.counts = 0
        self.all_correct = 0
        self.correct = None
        self.tp = None
        self.tn = None
        self.fp = None
        self.fn = None

    def update(self, logits, target):
        # batch x classes x time -> batch*time x classes
        logits = logits.permute(0,2,1).flatten(0, 1)
        target = target.permute(0,2,1).flatten(0, 1)
        mask = (target != combnet.MASK_INDEX).any(1)
        try:
            logits = logits[mask]
            target = target[mask]
        except:
            breakpoint()
        if self.correct is None:
            self.correct = torch.zeros(logits.shape[1]).to(target.device)
            self.tp = torch.zeros(logits.shape[1]).to(target.device)
            self.tn = torch.zeros(logits.shape[1]).to(target.device)
            self.fp = torch.zeros(logits.shape[1]).to(target.device)
            self.fn = torch.zeros(logits.shape[1]).to(target.device)
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        correct = preds == target
        true_positives = (preds == 1) & (target == 1)
        true_negaitves = (preds == 0) & (target == 0)
        false_negatives = (preds == 0) & (target == 1)
        false_positives = (preds == 1) & (target == 0) 
        self.tp += true_positives.sum(0)
        self.tn = true_negaitves.sum(0)
        self.fn += false_negatives.sum(0)
        self.fp += false_positives.sum(0)
        # all correct
        self.all_correct += correct.all(1).sum()
        # per class correct
        correct = correct.sum(0)
        self.correct += correct
        self.counts += target.shape[0]

    def __call__(self):
        result = {f'PerClassBinaryAccuracy/{idx}': (self.correct[idx]/self.counts).item() for idx in range(0, len(self.correct))}

        precision = self.tp / (self.tp + self.fp + 1e-9)
        result |= {f'PerClassPrecision/{idx}': (precision[idx]).item() for idx in range(0, len(precision))}

        recall = self.tp / (self.tp + self.fn + 1e-9)
        result |= {f'PerClassRecall/{idx}': (recall[idx]).item() for idx in range(0, len(recall))}

        F1 = 2*precision*recall / (precision+recall + 1e-9)
        result |= {f'PerClassF1/{idx}': (F1[idx]).item() for idx in range(0, len(F1))}

        macro_precision = precision.mean()
        result['MacroPrecision'] = macro_precision.item()

        macro_recall = recall.mean()
        result['MacroRecall'] = macro_recall.item()

        macro_F1 = F1.mean()
        result['MacroF1'] = macro_F1.item()

        tp = self.tp.sum()
        tn = self.tn.sum()
        fp = self.fp.sum()
        fn = self.fn.sum()

        micro_precision = tp / (tp + fp + 1e-9)
        result['MicroPrecision'] = micro_precision.item()

        micro_recall = tp / (tp + fn + 1e-9)
        result['MicroRecall'] = micro_recall.item()

        micro_F1 = 2*micro_precision*micro_recall / (micro_precision+micro_recall + 1e-9)
        result['MicroF1'] = micro_F1.item()

        result['TotalPerClassAcuracy'] = (self.correct.sum()/(self.counts*len(self.correct) + 1e-9)).item()
        result['FrameAccuracy'] = (self.all_correct/self.counts).item()
        return result
    
class HammingLoss(torchutil.metrics.Metric):

    def reset(self):
        self.loss = 0
        self.count = 0

    def update(self, logits, target):
        # batch x classes x time -> batch*time x classes
        logits = logits.permute(0,2,1).flatten(0, 1)
        target = target.permute(0,2,1).flatten(0, 1)
        mask = (target != combnet.MASK_INDEX).any(1)
        logits = logits[mask]
        target = target[mask]
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        loss = (preds != target).sum(1)
        self.loss += loss.sum()
        self.count += target.shape[0]

    def __call__(self):
        return (self.loss / self.count).item()