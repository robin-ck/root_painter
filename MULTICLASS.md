# Multiclass RootPainter with the pythonpanic model

This branch uses the RootPainter interface and corrective annotation workflow
with the land cover model from [pythonpanic](https://github.com/nvdeijk/pythonpanic)
(U-Net, ResNet34 encoder with ImageNet weights, ASPP bottleneck, sigmoid
output with one channel per class, BCE + Dice loss, Adam lr 1e-4).

## Workflow

1. **Make a dataset from GeoTIFFs**

   ```bash
   pip install rasterio numpy pillow
   python geo/split_geotiff.py path/to/maps --out ~/root_painter_sync/datasets/dommel --tile-size 900
   ```

   This writes PNG tiles, a `.pgw` world file per tile and `tiles_georef.json`
   (source map, CRS, affine transform and pixel offsets of every tile).

2. **Create a project** in the painter. The *Palette* section lists the
   classes. The defaults are forest, wetland, heather and water. You can rename,
   recolour, add or remove classes there. Background is always present.
   Brush colours are fixed once the project is created, because the trainer
   tells classes apart by colour.

3. **Annotate.** Shortcuts: `W` background, `1`-`9` classes (`Q` is also the
   first class), `E` eraser. A painted pixel means "this class and none of the
   others". Unpainted pixels are ignored during training.

4. **Train and segment** as usual. Segmentations show each class in its brush
   colour. The *Numpy Compressed (.npz)* output format stores the class index
   per pixel (0 = background, 1.. = classes in palette order).

## Notes

- Trainer dependency added: `segmentation-models-pytorch`. The first model in
  a project downloads ImageNet ResNet34 weights. Without internet it falls
  back to random weights.
- Models from the original RootPainter U-Net (`UNetGNRes`) can't be loaded by
  this trainer.
- Projects created before this change (no `classes` in the `.seg_proj` file)
  still work as a single-class project with red/green brushes.
- Per-class precision, recall and F1 are written to
  `logs/<date>_<train|cur_val|prev_val>_per_class.csv`. The `f1` in the
  existing CSVs is the mean F1 over classes, and a model is saved when this value improves.
- The measurement and conversion tools in the painter menus (length, region
  props, convert to annotation, metrics plot etc.) were not changed and
  assume a single foreground class.
