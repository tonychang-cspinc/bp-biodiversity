"""05-build-prediction-csv.py

Given a vector file, or an ecocode prefix labeled vector file and a set of ecocode prefixes, identify intersecting HLS tiles for those prefixes
Write the resulting tiles out to a csv.

Example:
python 05-build-prediction-csv.py \
    --vector-path="eco_us.geojson" \
    --ecocode-prefixes "-24" "M24" "M26" "-26" \
    --years 2015 2016 2017 2018 2019 \
    -o predict_24_26_m26_m24.csv
"""
import argparse
import io
from itertools import product

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box


def _get_extents():
    hls_tile_extents_url = 'https://ai4edatasetspublicassets.blob.core.windows.net/assets/S2_TilingSystem2-1.txt?st=2019-08-23T03%3A25%3A57Z&se=2028-08-24T03%3A25%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=KHNZHIJuVG2KqwpnlsJ8truIT5saih8KrVj3f45ABKY%3D'
    # Load this file into a table, where each row is:
    # Tile ID, Xstart, Ystart, UZ, EPSG, MinLon, MaxLon, MinLat, MaxLat
    print('Reading tile extents...')
    s = requests.get(hls_tile_extents_url).content
    hls_tile_extents = pd.read_csv(io.StringIO(s.decode('utf-8')),delimiter=r'\s+')
    print('Read tile extents for {} tiles'.format(len(hls_tile_extents)))
    return hls_tile_extents


def _get_boxes(tile_extents_df):
    return gpd.GeoDataFrame(
        tile_extents_df,
        crs="EPSG:4326",
        geometry=[
            box(r['MinLon'], r['MinLat'], r['MaxLon'], r['MaxLat'])
            for _, r in tile_extents_df.iterrows()
        ]
    )


def get_geometry_hls_tile_ids(hls_df, geom_df):
    tiles_in_aoi = gpd.overlay(geom_df, hls_df, how='intersection')
    return set(tiles_in_aoi['TilID'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector-path', help="Path to any valid, geopandas readable vector file")
    parser.add_argument('--ecocode-prefixes', nargs="+", default = [], help="Ecocode prefixes to find intersecting HLS tiles for")
    parser.add_argument('--years', nargs="+", help="HLS tile years to predict")
    parser.add_argument('-o', '--output-path', help="Path to file to save tiles + years csv to")
    args = parser.parse_args()
    if args.ecocode_prefixes:
        ecocode_df = gpd.read_file(args.vector_path)
        aoi_df = pd.concat([
            ecocode_df[ecocode_df.ECOCODE.str.match(prefix)]
            for prefix in args.ecocode_prefixes
        ])
    else:
        aoi_df = gpd.read_file(args.vector_path)
    hls_df = _get_boxes(_get_extents())
    tiles = get_geometry_hls_tile_ids(hls_df, aoi_df)
    pd.DataFrame(
        list(product(tiles, args.years)), columns=["tile", "year"]
    ).sort_values(
        ["tile", "year"], axis=0
    ).to_csv(args.output_path, index=False)
