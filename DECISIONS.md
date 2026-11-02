# Decisions: RootPainter interface + pythonpanic model

Goal: keep the RootPainter interface and corrective annotation workflow
([Abe404/root_painter](https://github.com/Abe404/root_painter)) and replace its
model with the multiclass land cover model from
[nvdeijk/pythonpanic](https://github.com/nvdeijk/pythonpanic).

Code: [robin-ck/root_painter, branch `multiclass-pythonpanic-model`](https://github.com/robin-ck/root_painter/tree/multiclass-pythonpanic-model)
(commit `974d84d`). Usage: [MULTICLASS.md](MULTICLASS.md).

## Decisions you made

| Question | Choice | Alternatives not chosen |
|---|---|---|
| Which RootPainter code to start from | **Current master**, turning on the existing `palette.py` brushes and rewriting the trainer for multiple classes | Abe404's `multiclass` branch (last updated March 2023, about 3 years behind master) |
| How classes behave when painting | **Exclusive + background**: one brush per class plus a background brush. A painted pixel means "this class and none of the others" | Independent layers per class (overlap allowed, more clicks). Softmax with one class per pixel (changes the model's behaviour) |
| Model and starting weights | **pythonpanic UNet-ASPP** (`model_aspp.py`, as used by `train.py`), **ImageNet encoder weights** | Start from one of your `.pth` checkpoints. Plain smp U-Net (`model.py`) |
| GeoTIFF / GeoPackage data | **Tile GeoTIFFs into a dataset**, keeping the georeference | Import gpkg labels as annotations. Export georeferenced GeoTIFF output (neither built) |
| Where to publish | **Fork in your GitHub account** (`robin-ck/root_painter`, public because it is a fork of a public repo). No pull request to Abe404 | Push to Abe404's repository (no access) |

## Design decisions that follow from these

### Model (trainer)
- **Architecture:** smp U-Net with a ResNet34 encoder and an ASPP block (dilation rates 6/12/18), copied from pythonpanic into `trainer/src/seg_model.py`.
- **Output:** one sigmoid channel per class and no background channel. A pixel is background when no class is above 0.5. Otherwise the most likely class wins.
- **Fitting RootPainter's tiles:** RootPainter uses 572 px input tiles and expects 500 px output tiles. A wrapper pads the input to a multiple of 32 and crops the centre, keeping the 36 px context margin on each side.
- **Loss:** pythonpanic's BCE + Dice, changed to ignore unpainted pixels rather than unlabelled classes per map. Dice only counts classes that appear in the batch.
- **Optimizer:** Adam with learning rate 1e-4, from pythonpanic. This replaces RootPainter's SGD 0.01 with momentum 0.99.
- **Model selection:** a new model is saved when the mean F1 over the classes on the validation set improves.
- **Image normalisation:** RootPainter's per-tile rescaling is kept, not pythonpanic's /255.
- **Old models:** old RootPainter models (`UNetGNRes`) can't be loaded.

### Annotations and classes
- **Storage:** the class list is saved in the `.seg_proj` file as `[name, [r, g, b, a], shortcut]`, with Background first.
- **Decoding:** annotations stay RGBA PNGs. The trainer assigns each painted pixel to the brush with the nearest colour.
- **Fixed colours:** brush colours can't change after the project is created, because the trainer tells classes apart by colour. The menu items to change brush colours are hidden for these projects.
- **Brush rules:** brush alpha is fixed at 180, so a brush can't look like the eraser. Names and colours must be unique, with at most 9 classes.
- **Default classes:** forest, wetland, heather, water, following pythonpanic.
- **Shortcuts:** `W` background, `1`–`9` classes, `Q` also selects the first class, `E` eraser.
- **Old projects:** projects without a class list keep the old red/green behaviour as a single class.

### Output
- **Segmentation PNG:** each class is shown in its brush colour.
- **`.npz` output:** class number per pixel (0 = background).
- **Per-class logs:** precision, recall and F1 per class go to new `*_per_class.csv` files. The existing CSV format is unchanged, and its `f1` column is the mean over classes.

### GeoTIFF tiling (`geo/split_geotiff.py`)
- **Tiles:** 900 px PNG tiles from bands 1–3 (RGB). The last tile in each row and column is shifted inward so all tiles are full size.
- **Georeference:** each tile gets a `.pgw` world file, and `tiles_georef.json` records the source map, CRS, affine transform and pixel offset of every tile.
- **Skipped tiles:** tiles without any valid pixels are left out.
- **Maps in the first dataset (`pythonpanic_maps`, 95 tiles):**
  - Dommel: `b568-1874`, `b586-1908`, `b588-1890`, `b629-1918`, `dommel_b627-1900`, `dommel_b628-1931`
  - IJssel: `b303-1871`, `ijssel_b320-1897`
- **Left out because they repeat other maps:** `b588-1890_1` and `_2` (the two halves of `b588-1890`), `train_map` and `val_map` (crops of other maps), and `maps/ijssel_b303-1871` (same map as `ijssel_geotiff/b303-1871`).

## Open points
- **Tile overlap:** the shifted edge tiles can overlap a lot. For example, in `b588-1890` the tiles starting at x=3600 and x=3801 share 699 px. Options are to skip a small leftover strip or to choose a tile size that fits the maps.
- **Nodata versus alpha:** the Dommel maps have nodata = 255 as well as an alpha band, so tiling goes by the nodata value. That made no difference for these maps, which have no transparent areas.
- **Not built:**
  - importing GeoPackage labels as annotations
  - putting segmentations back together into a georeferenced GeoTIFF
  - starting from an existing pythonpanic checkpoint
- **Still single-class:** the measurement and conversion tools in the painter, and the trainer's benchmark scripts (`test_training.py`, `sim_benchmark/`).
- **Packaging:** the PyInstaller and workstation builds weren't updated for the new `segmentation-models-pytorch` dependency.
- **Not yet tested:** a GPU and real training runs on these maps. So far there have only been tests and a run on synthetic data, where class F1 reached 0.95–0.99.
