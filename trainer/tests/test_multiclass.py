"""
Tests for multiclass training with the pythonpanic U-Net ASPP model.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
import os
import sys
import json

import numpy as np
import pytest
import torch
from skimage.io import imsave, imread

# Add trainer/src to sys.path so we can import the project modules
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src')
sys.path.insert(0, src_dir)

import im_utils
import model_utils
from loss import masked_bce_dice_loss
from metrics import class_confusion, get_multiclass_metrics
from seg_model import create_model, num_classes_from_state_dict
from trainer import Trainer

CLASSES = [
    ['Background', [0, 255, 0, 180], 'W'],
    ['forest', [255, 0, 0, 180], '1'],
    ['wetland', [255, 0, 255, 180], '2'],
    ['water', [0, 0, 255, 180], '3'],
]


def test_annot_to_onehot_nearest_colour():
    annot = np.zeros((4, 4, 4), dtype=np.uint8)
    annot[0, 0] = [1, 254, 0, 180]    # background, slightly off colour
    annot[0, 1] = [254, 0, 1, 180]    # forest
    annot[0, 2] = [255, 1, 254, 180]  # wetland
    annot[0, 3] = [0, 0, 255, 180]    # water
    onehot = im_utils.annot_to_onehot(annot, CLASSES)
    assert onehot.shape == (4, 4, 4)
    assert list(np.argmax(onehot[0], axis=1)) == [0, 1, 2, 3]
    # unpainted pixels are undefined
    assert onehot[1:].sum() == 0


def test_annot_to_onehot_legacy():
    annot = np.zeros((2, 2, 4), dtype=np.uint8)
    annot[0, 0] = [255, 0, 0, 180]
    annot[0, 1] = [0, 255, 0, 180]
    onehot = im_utils.annot_to_onehot(annot)
    assert onehot[0, 0].tolist() == [0, 1]
    assert onehot[0, 1].tolist() == [1, 0]
    assert onehot[1].sum() == 0


def test_model_output_matches_tile_scheme():
    model = create_model(3, pretrained=False)
    model.eval()
    for in_w in [572, 252]:
        out = model(torch.zeros(1, 3, in_w, in_w))
        assert out.shape == (1, 3, in_w - 72, in_w - 72)
    assert num_classes_from_state_dict(model.state_dict()) == 3


def test_loss_ignores_undefined_pixels():
    logits = torch.zeros(1, 2, 8, 8, requires_grad=True)
    targets = torch.zeros(1, 2, 8, 8)
    targets[:, 0, :4] = 1
    defined = torch.zeros(1, 8, 8)
    defined[:, :4] = 1
    loss = masked_bce_dice_loss(logits, targets, defined)
    loss.backward()
    # no gradient where nothing was annotated
    assert logits.grad[:, :, 4:].abs().sum() == 0
    assert logits.grad[:, :, :4].abs().sum() > 0


def test_multiclass_metrics():
    true = np.array([[0, 1, 1, 2]])
    pred = np.array([[0, 1, 2, 2]])
    defined = np.ones_like(true)
    counts = class_confusion(pred, true, defined, 3)
    assert counts[0].tolist() == [1, 0, 2, 1]   # class 1: tp fp tn fn
    assert counts[1].tolist() == [1, 1, 2, 0]   # class 2
    m = get_multiclass_metrics(counts, 4, 0.0)
    # class 3 is absent so only classes 1 and 2 count
    assert m['f1'] == pytest.approx((2/3 + 2/3) / 2)


def make_project(sync_dir, n_images=2, size=300):
    """ images with a red, magenta and blue square on grey, annotated with
        the matching brush colours """
    rng = np.random.default_rng(0)
    dataset = sync_dir / 'datasets' / 'maps'
    proj = sync_dir / 'projects' / 'maps'
    for d in [dataset, proj / 'annotations' / 'train', proj / 'annotations' / 'val',
              proj / 'segmentations', proj / 'models', proj / 'messages', proj / 'logs']:
        d.mkdir(parents=True)
    for i in range(n_images):
        im = np.full((size, size, 3), 128, dtype=np.uint8)
        im += rng.integers(0, 20, im.shape, dtype=np.uint8)
        annot = np.zeros((size, size, 4), dtype=np.uint8)
        annot[:40, :] = CLASSES[0][1]
        for c, (y, x) in enumerate([(60, 60), (60, 180), (180, 100)], start=1):
            im[y:y+80, x:x+80] = CLASSES[c][1][:3]
            annot[y+10:y+70, x+10:x+70] = CLASSES[c][1]
        imsave(dataset / f'im{i}.png', im, check_contrast=False)
        split = 'train' if i == 0 else 'val'
        imsave(proj / 'annotations' / split / f'im{i}.png', annot,
               check_contrast=False)
    return dataset, proj


def test_train_validate_and_segment(tmp_path):
    for folder in ['instructions', 'executed_instructions',
                   'failed_instructions', 'projects', 'datasets']:
        (tmp_path / folder).mkdir()
    dataset, proj = make_project(tmp_path)
    trainer = Trainer(sync_dir=tmp_path, patch_size=252, max_workers=0)
    trainer.bs = 2
    config = {
        'model_dir': 'projects/maps/models',
        'dataset_dir': 'datasets/maps',
        'train_annot_dir': 'projects/maps/annotations/train',
        'val_annot_dir': 'projects/maps/annotations/val',
        'seg_dir': 'projects/maps/segmentations',
        'log_dir': 'projects/maps/logs',
        'message_dir': 'projects/maps/messages',
        'classes': CLASSES,
    }
    trainer.start_training(trainer.fix_config_paths(config))
    assert trainer.train_config['classes'] == CLASSES
    trainer.train_set.min_epoch_tiles = 4
    trainer.train_one_epoch()

    logs = os.listdir(proj / 'logs')
    assert any(f.endswith('_cur_val_per_class.csv') for f in logs)
    per_class = [f for f in logs if f.endswith('_train_per_class.csv')][0]
    header = open(proj / 'logs' / per_class).readline()
    assert 'forest_f1' in header and 'water_f1' in header

    seg_config = trainer.fix_config_paths({
        'dataset_dir': 'datasets/maps',
        'seg_dir': 'projects/maps/segmentations',
        'model_dir': 'projects/maps/models',
        'file_names': ['im1.png'],
        'classes': CLASSES,
    })
    trainer.segment(seg_config)
    seg = imread(proj / 'segmentations' / 'im1.png')
    assert seg.shape == (300, 300, 4)
    # only background (transparent) or brush colours
    colours = {tuple(c[1][:3]) for c in CLASSES[1:]}
    painted = seg[seg[:, :, 3] > 0][:, :3]
    assert all(tuple(p) in colours for p in np.unique(painted, axis=0))

    # npz output has class indices
    seg_config['format'] = 'Numpy Compressed (.npz)'
    trainer.segment(seg_config)
    npz = np.load(proj / 'segmentations' / 'im1.npz')['seg']
    assert npz.shape == (300, 300) and npz.max() <= 3
