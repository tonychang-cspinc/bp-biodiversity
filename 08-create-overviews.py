"""
Create overview TIFFs + mosaic json for each z6 tile for a mosaicjson.

Inspired by https://github.com/kylebarron/naip-cogeo-mosaic/blob/master/code/overviews.py

Example run for each of your variables/years (e.g. canopy_cvr, basal_area, bio_acre; 2016, 2018) e.g.:
python 09-create-overviews.py \
-z 6 \
-m outputs/canopy_cvr-2018-z8-z14.json \
-o overviews/canopy_cvr/2018 \
-p "outputs/canopy_cvr-2018" \
-c "https://usfs.blob.core.windows.net/app/overview_cog/canopy_cvr/2018" \
--min-zoom=6 \
--max-zoom=7 \
-t 1024

"""
import argparse
import concurrent.futures
import itertools
import json
import os
from os.path import exists
from typing import Dict
from typing import List
from typing import Set

import fsspec
import mercantile
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from rio_tiler.io import COGReader
from rio_tiler.models import ImageData
from rio_tiler.mosaic import mosaic_reader

TILE_SIZE = 1024
NO_DATA = -1


def tiler(src_path: str, *args, **kwargs) -> ImageData:
    with COGReader(src_path, nodata=NO_DATA, resampling_method='nearest') as cog:
        return cog.tile(*args, **kwargs)


def get_asset_list(mosaic: Dict, qk: str) -> List[str]:
    if len(qk) < mosaic['quadkey_zoom']:
        return list(itertools.chain.from_iterable(
            assets
            for aqk, assets in mosaic['tiles'].items()
            if aqk.startswith(qk)
        ))
    else:
        for aqk, assets in mosaic['tiles'].items():
            if qk.startswith(aqk):
                return assets


def get_tiles(mosaic: Dict, zoom: int) -> Set[mercantile.Tile]:
    """Given a mosaicjson dict and a zoom, find all quadkeys at zoom level `zoom` that have data per mosaicjson"""
    assert zoom <= mosaic['quadkey_zoom']
    return set(
        mercantile.quadkey_to_tile(qk[:zoom])
        for qk in mosaic['tiles'].keys()
    )


def save_tile(tile, img, output_dir, size):
    """Save on-disk Cloud-optimized geotiffs (COG) one for each band
    """
    profile = dict(
        driver="GTiff",
        dtype="float32",
        count=1,
        height=size,
        width=size,
        crs=img.crs,
        transform=img.transform,
        nodata=NO_DATA
    )
    with MemoryFile() as memfile:
        with memfile.open(**profile) as mem:
            mem.write(img.data)
            dst_profile = cog_profiles.get("deflate")
            cog_translate(
                mem,
                f"{output_dir}/{mercantile.quadkey(tile)}.tif",
                dst_profile,
                in_memory=True,
                quiet=True,
            )


def create_mosaic_json(mosaic, zoom, min_zoom, max_zoom, tiles, prefix, cog_prefix):
    base_dict = {
        "mosaicjson": "0.0.2",
        "name": mosaic['name'],
        "description": mosaic['description'],
        "version": "1.0.0",
        "attribution": "Conservation Science Partners",
        "minzoom": min_zoom,
        "maxzoom": max_zoom,
        "quadkey_zoom": zoom,
        "bounds": [-180, -90, 180, 90],
        "tiles": {mercantile.quadkey(t): [f"{cog_prefix}/{mercantile.quadkey(t)}.tif"] for t in tiles}
    }
    with open(f'{prefix}-overview-z{min_zoom}-z{max_zoom}.json', 'w') as f:
        print(f'{prefix}-overview-z{min_zoom}-z{max_zoom}.json')
        json.dump(base_dict, f)


def get_and_save_tile(assets, tile, i, output_dir, tilesize):
    if exists(f"{output_dir}/{mercantile.quadkey(tile)}.tif"):
        print(f"Skipping {mercantile.quadkey(tile)}")
        return 0
    img, _ = mosaic_reader(
        assets,
        tiler,
        tile.x,
        tile.y,
        tile.z,
        tilesize=tilesize
    )
    save_tile(tile, img, output_dir, tilesize)
    print(f"Finished {mercantile.quadkey(tile)}, {i+1}/{len(tiles)}")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--mosaic-path',
        help='URL or path to mosaicjson file to create overviews from'
    )
    parser.add_argument('-z', '--zoom', type=int, help="zoom level to create overviews at")
    parser.add_argument('--min-zoom', type=int, help="min zoom level for output mosaic")
    parser.add_argument('--max-zoom', type=int, help="max zoom level for output mosaic")
    parser.add_argument('-o', '--output-dir', help="Path to write output overview COGs to")
    parser.add_argument('-p', '--mosaic-prefix', help="What is the prefix (including path) for the output mosaic")
    parser.add_argument('-c', '--cog-prefix', help="What will the url for the resulting COGs be")
    parser.add_argument('-t', '--tile-size', type=int, help="How large should each tile be in pixels")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with fsspec.open(args.mosaic_path, 'r') as f:
        mosaic = json.load(f)
    tiles = get_tiles(mosaic, args.zoom)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        res = executor.map(
            get_and_save_tile,
            *zip(*list(
                (get_asset_list(mosaic, mercantile.quadkey(t)), t, i, args.output_dir, args.tile_size)
                for i, t in enumerate(tiles)
            ))
        )

    create_mosaic_json(mosaic, args.zoom, args.min_zoom, args.max_zoom, tiles, args.mosaic_prefix, args.cog_prefix)
