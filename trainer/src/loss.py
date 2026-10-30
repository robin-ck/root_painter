"""
Copyright (C) 2019 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from torch.nn.functional import binary_cross_entropy
from torch.nn.functional import binary_cross_entropy_with_logits


def dice_loss(predictions, labels):
    """ based on loss function from V-Net paper """
    softmaxed = softmax(predictions, 1)
    predictions = softmaxed[:, 1, :]  # just the root probability.
    labels = labels.float()
    preds = predictions.contiguous().view(-1)
    labels = labels.view(-1)
    intersection = torch.sum(torch.mul(preds, labels))
    union = torch.sum(preds) + torch.sum(labels)
    return 1 - ((2 * intersection) / (union))


def combined_loss(predictions, labels):
    """ mix of dice and cross entropy """
    # if they are bigger than 1 you get a strange gpu error
    # without a stack track so you will have no idea why.
    assert torch.max(labels) <= 1
    if torch.sum(labels) > 0:
        return (dice_loss(predictions, labels) +
                (0.3 * cross_entropy(predictions, labels)))
    # When no roots use only cross entropy
    # as dice is undefined.
    return 0.3 * cross_entropy(predictions, labels)



def dice_loss2(preds, labels):
    """ based on loss function from V-Net paper """
    assert torch.max(labels) <= 1
    assert torch.min(labels) >= 0
    assert torch.max(preds) <= 1
    assert torch.min(preds) >= 0

    intersection = torch.sum(torch.mul(preds, labels))
    union = torch.sum(preds) + torch.sum(labels)
    return 1 - ((2 * intersection) / (union))


def combined_loss2(preds, labels, mask=None):
    """ mix of dice and BCE for single-channel sigmoid output.
        Not yet used - needs benchmarking against combined_loss before switching. """
    if mask is not None:
        preds = torch.mul(preds, mask)
    assert torch.max(labels) <= 1
    cx = 0.3 * binary_cross_entropy(preds, labels)
    if torch.sum(labels) > 0:
        return dice_loss2(preds, labels) + cx
    return cx


def masked_bce_dice_loss(logits, targets, defined):
    """ Multiclass loss from pythonpanic (source/losses.py): BCE + Dice on
        sigmoid outputs, one channel per foreground class.

        logits, targets: (B, C, H, W). targets are 1 for the painted class
        and 0 for the other classes (background pixels are 0 everywhere).
        defined: (B, H, W), 1 where the pixel was annotated. Pixels that
        were not annotated do not count towards the loss.
    """
    defined = defined[:, None].expand_as(logits)
    bce = binary_cross_entropy_with_logits(logits, targets, reduction='none')
    bce = (bce * defined).sum() / defined.sum().clamp_min(1.0)

    probs = torch.sigmoid(logits) * defined
    tp = (probs * targets).sum(dim=(0, 2, 3))
    fp = (probs * (1 - targets)).sum(dim=(0, 2, 3))
    fn = ((1 - probs) * targets * defined).sum(dim=(0, 2, 3))
    eps = 1e-6
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    # Dice is only meaningful for classes present in the annotation
    class_present = targets.sum(dim=(0, 2, 3)) > 0
    if class_present.any():
        return bce + (1 - dice[class_present]).mean()
    return bce
