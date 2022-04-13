"""
Script for generating mosaicjson files for use with titiler

Usage:
python 07-create-mosaicjson.py \
--base-path="https://usfs.blob.core.windows.net/app/hls_cog" \
--tiles-path="tiles.csv" \
-y 2016 2018 \
-v class canopy_cvr basal_area bio_acre \
--min-zoom=8 \
--max-zoom=14

Note:
This can just be replaced by cogeo-mosaic
e.g.
cat cogs.txt | cogeo-mosaic create --minzoom=8 --maxzoom=14 --quadkey-zoom=8 --threads=8 -o mymosaic.json -

where cogs.txt is a text file with one cog url per line

"""
import argparse
import json
import os
import tempfile
import urllib.request
from collections import defaultdict


import mercantile
import pandas as pd


temp_dir = os.path.join(tempfile.gettempdir())
os.makedirs(temp_dir, exist_ok=True)

BASE_URL = "{base_path}/{variable}/{year}/{tile}.tif"


def generate_mosaic_json(year, variable, min_zoom, max_zoom, hls_tiles, base_path, output_dir):
    dd = defaultdict(list)  # quadkey -> list of COGs intersecting quadkey
    for hls_tile_id, row in extents.iterrows():
        if hls_tile_id not in hls_tiles:
            continue
        for tile in mercantile.tiles(row.MinLon, row.MinLat, row.MaxLon, row.MaxLat, zooms=min_zoom):
            dd[mercantile.quadkey(tile)].append(BASE_URL.format(base_path=base_path, variable=variable, year=year, tile=hls_tile_id))
    base_dict = {
        "mosaicjson": "0.0.2",
        "name": f"{variable} {year}",
        "description": f"{variable} {year}",
        "version": "1.0.0",
        "attribution": "Conservation Science Partners",
        "minzoom": min_zoom,
        "maxzoom": max_zoom,
        "quadkey_zoom": min_zoom,
        "bounds": [-180, -90, 180, 90],
        "tiles": dd
    }
    with open(f'{output_dir}/{variable}-{year}-z{min_zoom}-z{max_zoom}.json', 'w') as f:
        json.dump(base_dict, f)


def download_url(url, destination_filename=None, progress_updater=None, force_download=False):
    """
    Download a URL to a temporary file
    """

    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    if destination_filename is None:
        url_as_filename = url.replace('://', '_').replace('/', '_')
        destination_filename = \
            os.path.join(temp_dir,url_as_filename)
    if (not force_download) and (os.path.isfile(destination_filename)):
        print('Bypassing download of already-downloaded file {}'.format(
            os.path.basename(url)))
        return destination_filename
    print('Downloading file {} to {}'.format(os.path.basename(url),
                                             destination_filename),end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)
    assert(os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    print('...done, {} bytes.'.format(nBytes))
    return destination_filename


def _get_hls_extents():
    hls_tile_extents_url = 'https://ai4edatasetspublicassets.blob.core.windows.net/assets/S2_TilingSystem2-1.txt?st=2019-08-23T03%3A25%3A57Z&se=2028-08-24T03%3A25%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=KHNZHIJuVG2KqwpnlsJ8truIT5saih8KrVj3f45ABKY%3D'
    # Load this file into a table, where each row is:
    # Tile ID, Xstart, Ystart, UZ, EPSG, MinLon, MaxLon, MinLat, MaxLat
    print('Reading HLS tile extents...')
    fp = temp_dir + '/hls_extents.csv'
    download_url(hls_tile_extents_url, fp)
    # s = requests.get(hls_tile_extents_url).content
    hls_tile_extents = pd.read_csv(fp, delimiter=r'\s+').set_index('TilID')
    print('Read HLS tile extents for {} tiles'.format(len(hls_tile_extents)))
    return hls_tile_extents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--output-dir',
        default='outputs',
        help='Location to save output mosaicjsons to'
    )
    parser.add_argument('--base-path', help="Base path to prediction tiles (e.g. blob storage url or local path)")
    parser.add_argument('--tiles-path', help="Path to CSV with a `tiles` column, each row is a name of an HLS tile (e.g. 10TET)")
    parser.add_argument('-y', '--years', type=int, nargs='+', help='years to generate')
    parser.add_argument('-v', '--variables', type=str, nargs='+', help='variables to generate')
    parser.add_argument('--min-zoom', type=int, help='min-zoom')
    parser.add_argument('--max-zoom', type=int, help='max-zoom')
    args = parser.parse_args()
    extents = _get_hls_extents()
    tile_list = pd.read_csv(args.tiles_path)['tile'].values.tolist()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for year in args.years:
        for var in args.variables:
            generate_mosaic_json(year, var, args.min_zoom, args.max_zoom, tile_list, args.base_path, args.output_dir)
