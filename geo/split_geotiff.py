"""
Cut large GeoTIFF maps into PNG tiles that can be used as a RootPainter
dataset, and record where each tile came from so segmentations can be put
back on the map later.

Usage:
    python split_geotiff.py MAP_OR_FOLDER [MAP_OR_FOLDER ...] \
        --out ~/root_painter_sync/datasets/my_maps --tile-size 900

For every tile a PNG is written to the output folder, plus a world file
(.pgw) so the tile opens in the right place in a GIS, and one
tiles_georef.json describing all tiles:

    {"<tile>.png": {"source": "...tif", "crs": "<WKT>",
                    "transform": [a, b, c, d, e, f],   # affine of the tile
                    "col_off": 0, "row_off": 0, "width": 900, "height": 900}}

Only the first three bands (RGB) are used, as in pythonpanic. Tiles with no
valid pixels (nodata / alpha 0 everywhere) are skipped.

Requires rasterio, numpy and Pillow.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image


def tile_starts(full_size, tile_size):
    """ Start offsets that cover the whole axis with full size tiles.
        The last tile is moved inward (overlapping its neighbour) instead
        of being a thin sliver at the edge. """
    if full_size <= tile_size:
        return [0]
    starts = list(range(0, full_size - tile_size, tile_size))
    starts.append(full_size - tile_size)
    return starts


def to_uint8(rgb, src_min, src_max):
    if rgb.dtype == np.uint8:
        return rgb
    scale = max(float(src_max - src_min), 1e-6)
    rgb = (rgb.astype(np.float32) - src_min) / scale
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


def valid_mask(src, window):
    """ True where the tile has data, from the dataset mask
        (covers nodata values and alpha bands). """
    masks = src.read_masks([1, 2, 3], window=window)
    return np.any(masks > 0, axis=0)


def split_map(map_path, out_dir, tile_size, index):
    with rasterio.open(map_path) as src:
        if src.count < 3:
            raise ValueError(f'Expected at least 3 bands in {map_path}, '
                             f'found {src.count}')
        if src.dtypes[0] != 'uint8':
            # one range for the whole map so tiles stay comparable
            sample = src.read([1, 2, 3], out_shape=(3, min(src.height, 2000),
                                                    min(src.width, 2000)))
            src_min, src_max = np.percentile(sample, [0.5, 99.5])
        else:
            src_min = src_max = None
        crs = src.crs.to_wkt() if src.crs else None
        stem = Path(map_path).stem
        written = 0
        for row_off in tile_starts(src.height, tile_size):
            for col_off in tile_starts(src.width, tile_size):
                window = Window(col_off, row_off,
                                min(tile_size, src.width - col_off),
                                min(tile_size, src.height - row_off))
                if not valid_mask(src, window).any():
                    continue
                rgb = src.read([1, 2, 3], window=window)
                rgb = to_uint8(rgb, src_min, src_max)
                fname = f'{stem}_r{row_off}_c{col_off}.png'
                Image.fromarray(np.moveaxis(rgb, 0, -1)).save(
                    os.path.join(out_dir, fname))

                t = src.window_transform(window)
                # world file: x pixel size, rotation, rotation, y pixel size,
                # then the centre of the top left pixel
                with open(os.path.join(out_dir, fname[:-4] + '.pgw'), 'w') as f:
                    f.write('\n'.join(str(v) for v in [
                        t.a, t.d, t.b, t.e,
                        t.c + t.a / 2 + t.b / 2,
                        t.f + t.d / 2 + t.e / 2]) + '\n')

                index[fname] = {
                    'source': os.path.abspath(map_path),
                    'crs': crs,
                    'transform': [t.a, t.b, t.c, t.d, t.e, t.f],
                    'col_off': int(col_off),
                    'row_off': int(row_off),
                    'width': int(window.width),
                    'height': int(window.height),
                }
                written += 1
    print(f'{map_path}: {written} tiles')


def find_maps(paths):
    maps = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            maps += sorted(f for f in p.iterdir()
                           if f.suffix.lower() in ('.tif', '.tiff'))
        else:
            maps.append(p)
    return maps


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('maps', nargs='+', help='GeoTIFF files or folders of them')
    parser.add_argument('--out', required=True,
                        help='output folder, usually inside <sync dir>/datasets')
    parser.add_argument('--tile-size', type=int, default=900,
                        help='tile width and height in pixels (default 900)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, 'tiles_georef.json')
    index = {}
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
    for map_path in find_maps(args.maps):
        split_map(map_path, args.out, args.tile_size, index)
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=1)
    print(f'{len(index)} tiles described in {index_path}')


if __name__ == '__main__':
    main()
