"""Supervised optimization and held-out evaluation; no test-based selection."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from utils import classification_metrics


class FocalLoss(nn.Module):
    """Unweighted multiclass focal loss, gamma=0 exactly recovers mean CE.

    No alpha/class weighting: only the hard-example modulation is changed.
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        if gamma < 0:
            raise ValueError('gamma must be nonnegative')
        self.gamma = float(gamma)

    def forward(self, logits, labels):
        log_pt = F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
        # expm1 is stable even when p_t is close to one.
        return ((-torch.expm1(log_pt)).pow(self.gamma) * (-log_pt)).mean()


def train_one_epoch(model, criterion, data_loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    for data in data_loader:
        bag = data['input'].to(device, dtype=torch.float32, non_blocking=True)
        labels = data['label'].to(device, dtype=torch.long, non_blocking=True)
        mask = data['mask'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(bag, mask=mask)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        n = len(labels)
        total_loss += float(loss.detach()) * n
        total_correct += int((logits.argmax(-1) == labels).sum())
        total_examples += n
    if total_examples == 0:
        raise ValueError('Empty training split')
    return {'loss': total_loss / total_examples, 'acc': 100 * total_correct / total_examples}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf, return_predictions=False):
    model.eval()
    total_loss, total_ce, total_examples = 0.0, 0.0, 0
    probabilities_all, labels_all, slide_ids = [], [], []
    for data in data_loader:
        bag = data['input'].to(device, dtype=torch.float32, non_blocking=True)
        labels = data['label'].to(device, dtype=torch.long, non_blocking=True)
        mask = data['mask'].to(device, non_blocking=True)
        logits = model(bag, mask=mask)
        loss = criterion(logits, labels)
        probabilities = F.softmax(logits, dim=-1)
        n = len(labels)
        total_loss += float(loss) * n
        # Also save CE for comparisons: focal loss values are on another scale.
        total_ce += float(F.cross_entropy(logits, labels)) * n
        total_examples += n
        probabilities_all.append(probabilities.cpu())
        labels_all.append(labels.cpu())
        slide_ids.extend(data['slide_id'])
    if total_examples == 0:
        raise ValueError('Empty evaluation split')
    probs, labels = torch.cat(probabilities_all), torch.cat(labels_all)
    metrics = classification_metrics(probs, labels, conf.n_class)
    metrics.update(loss=total_loss / total_examples, ce_loss=total_ce / total_examples)
    if return_predictions:
        return metrics, {'slide_id': slide_ids, 'labels': labels.numpy(), 'probabilities': probs.numpy()}
    return metrics
