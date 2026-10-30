"""
Computing metrics on output segmentations for root images

Copyright (C) 2019, 2020 Abraham George Smith

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

# pylint: disable=C0111,R0913
from datetime import datetime
import numpy as np

def get_metrics_str(all_metrics, to_use=None):
    out_str = ""
    for name, val in all_metrics.items():
        if to_use is None or name in to_use:
            out_str += f" {name} {val:.4g}"
    return out_str

def get_metric_csv_row(metrics):
    now_str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    parts = [now_str, metrics['TP'], metrics['FP'], metrics['TN'],
             metrics['FN'], round(metrics['precision'], 4),
             round(metrics['recall'], 4), round(metrics['f1'], 4),
             metrics['defined'], metrics['duration'], metrics['loss']]
    return ','.join([str(p) for p in parts]) + '\n'


def get_metrics(tp, fp, tn, fn, defined_sum, duration, loss=float('nan')):
    total = (tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    assert not np.isnan(tp)
    assert not np.isnan(fp)
    assert not np.isnan(fn)
    assert not np.isnan(fp)
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
        iou = tp / (tp + fp + fn)
    else:
        precision = recall = f1 = iou = float('NaN')
    return {
        "accuracy": accuracy,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "true_mean": (tp + fn) / total,
        "pred_mean": (tp + fp) / total,
        "defined": defined_sum,
        "duration": duration,
        "loss": loss,
    }


def class_confusion(pred_map, true_map, defined, num_classes):
    """ Per foreground class TP, FP, TN, FN counted over defined pixels.
        pred_map and true_map are class index maps (0 = background).
        Works for numpy arrays and torch tensors.
        Returns a (num_classes, 4) numpy array of [tp, fp, tn, fn]. """
    pred = pred_map[defined > 0]
    true = true_map[defined > 0]
    counts = np.zeros((num_classes, 4), dtype=np.int64)
    for c in range(1, num_classes + 1):
        p = pred == c
        t = true == c
        counts[c-1] = [int((p & t).sum()), int((p & ~t).sum()),
                       int((~p & ~t).sum()), int((~p & t).sum())]
    return counts


def get_multiclass_metrics(counts, defined_sum, duration, loss=float('nan')):
    """ Summed TP/FP/TN/FN over classes plus 'f1' as the mean F1 of the
        classes that appear in the annotation or prediction (macro F1),
        which is used to decide whether a model is better.
        Per class metrics are in 'per_class'. """
    tp, fp, tn, fn = [int(v) for v in counts.sum(axis=0)]
    metrics = get_metrics(tp, fp, tn, fn, defined_sum, duration, loss)
    per_class = [get_metrics(*[int(v) for v in c], defined_sum, duration)
                 for c in counts]
    f1s = []
    for c, m in zip(counts, per_class):
        c_tp, c_fp, _, c_fn = c
        if c_tp + c_fp + c_fn > 0:
            f1s.append(0.0 if np.isnan(m['f1']) else m['f1'])
    metrics['f1'] = float(np.mean(f1s)) if f1s else float('nan')
    metrics['per_class'] = per_class
    return metrics


def get_per_class_csv_row(metrics):
    now_str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    parts = [now_str]
    for m in metrics['per_class']:
        parts += [round(m['precision'], 4), round(m['recall'], 4),
                  round(m['f1'], 4)]
    return ','.join([str(p) for p in parts]) + '\n'


def get_per_class_csv_header(class_names):
    cols = ['date_time']
    for name in class_names:
        cols += [f'{name}_precision', f'{name}_recall', f'{name}_f1']
    return ','.join(cols)
