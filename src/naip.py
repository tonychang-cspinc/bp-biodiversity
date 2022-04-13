# Standard packages
import io
import os
import re
import tempfile
import urllib
from collections import defaultdict

# Less standard, but still pip- or conda-installable
import fiona.transform
import numpy as np
import requests
import rtree
import shapely
import pandas as pd
import pickle
import xarray as xr

from affine import Affine

latest_wkid = 3857
CRS = "EPSG:4326"

# Storage locations are documented at http://aka.ms/ai4edata-naip

# NAIP in the East US Azure region
NAIP_ROOT = 'https://naipblobs.blob.core.windows.net/naip'

index_files = ["tile_index.dat", "tile_index.idx", "tiles.p"]
index_blob_root = re.sub('/naip$', '/naip-index/rtree/', NAIP_ROOT)
temp_dir = os.path.join(tempfile.gettempdir(),'naip')
os.makedirs(temp_dir,exist_ok=True)


class NAIPTileIndex:
    """
    Utility class for performing NAIP tile lookups by location.
    """

    tile_rtree = None
    tile_index = None
    base_path = None

    def __init__(self, base_path=None):

        if base_path is None:

            base_path = temp_dir
            os.makedirs(base_path,exist_ok=True)

            for file_path in index_files:
                download_url(index_blob_root + file_path, base_path + '/' + file_path)

        self.base_path = base_path
        self.tile_rtree = rtree.index.Index(base_path + "/tile_index")
        self.tile_index = pickle.load(open(base_path  + "/tiles.p", "rb"))
        self.hls_extents = _get_hls_extents()

    def lookup_tile(self, lat, lon):
        """"
        Given a lat/lon coordinate pair, return the list of NAIP tiles that contain
        that location.

        Returns a list of COG file paths.
        """

        point = shapely.geometry.Point(float(lon),float(lat))
        intersected_indices = list(self.tile_rtree.intersection(point.bounds))

        intersected_files = []
        tile_intersection = False

        for idx in intersected_indices:

            intersected_file = self.tile_index[idx][0]
            intersected_geom = self.tile_index[idx][1]
            if intersected_geom.contains(point):
                tile_intersection = True
                intersected_files.append(intersected_file)

        if not tile_intersection and len(intersected_indices) > 0:
            print('''Error: there are overlaps with tile index,
                      but no tile completely contains selection''')
            return None
        elif len(intersected_files) <= 0:
            print("No tile intersections")
            return None
        else:
            return intersected_files

    def intersect_hls_tile(self, hls_tile, year):
        row = self.hls_extents.loc[self.hls_extents['TilID'] == hls_tile].iloc[0]
        bbox = [row.MinLon, row.MinLat, row.MaxLon, row.MaxLat]
        idxs = self.tile_rtree.intersection(bbox)
        urls = {self.tile_index[idx][0] for idx in idxs}
        id_to_urls = defaultdict(list)
        for url in urls:
            fname = url.split('/')[-1]
            match = re.search(r'^[A-z]_(?P<id>[0-9]{7}_[A-z]{2}_[0-9]{2})_.*\.tif$', fname)
            id_to_urls[match.group('id')].append(url)
        closest_urls = []
        for k, vs in id_to_urls.items():
            years = []
            for url in vs:
                fname = url.split('/')[-1]
                match = re.search(r'^.*?(?P<year>[0-9]{4})[0-9]{4}.*\.tif$', fname)
                years.append(int(match.group('year')))
            arr = np.array(sorted(years))
            idx = (np.abs(arr - year)).argmin()
            closest_urls.append(f"{NAIP_ROOT}/{vs[years.index(arr[idx])]}")
        return sorted(closest_urls)


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
        print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url)))
        return destination_filename
    print('Downloading file {} to {}'.format(os.path.basename(url),destination_filename),end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)
    assert(os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    print('...done, {} bytes.'.format(nBytes))
    return destination_filename


def get_naip_chip(lat, lon, year, chip_size, index):
    url = get_naip_url(lat, lon, year, index)
    half_chip = int(chip_size/2)
    da = xr.open_rasterio(url)

    naip_tfm = Affine(*da.attrs['transform'])
    # Convert our lat/lon point to the local NAIP coordinate system
    x_tile_crs, y_tile_crs = fiona.transform.transform(
        CRS, da.attrs['crs'], [lon], [lat]
    )
    x_tile_crs = x_tile_crs[0]
    y_tile_crs = y_tile_crs[0]

    # Convert our new x/y coordinates into pixel indices
    x_tile_offset, y_tile_offset = ~naip_tfm * (x_tile_crs, y_tile_crs)
    x_tile_offset = int(np.floor(x_tile_offset))
    y_tile_offset = int(np.floor(y_tile_offset))

    padded = da.pad({'x': half_chip, 'y': half_chip}, mode='reflect')
    return padded.isel(
        {
            'x': list(range(x_tile_offset-half_chip, x_tile_offset+half_chip)),
            'y': list(range(y_tile_offset-half_chip, y_tile_offset+half_chip))
        }
    ).transpose('band', 'x', 'y')[:3, :, :].compute().values / 255


def get_naip_url(lat, lon, year, index):
    naip_files = index.lookup_tile(lat, lon)
    if naip_files:
        naip_years = np.array([int(n.split("/")[2]) for n in naip_files])
        closest = min(naip_years, key=lambda x: abs(x - year))
        match_idx = np.where(naip_years == closest)[0][0]
        return f"{NAIP_ROOT}/{naip_files[match_idx]}"


def _get_hls_extents():
    hls_tile_extents_url = 'https://ai4edatasetspublicassets.blob.core.windows.net/assets/S2_TilingSystem2-1.txt?st=2019-08-23T03%3A25%3A57Z&se=2028-08-24T03%3A25%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=KHNZHIJuVG2KqwpnlsJ8truIT5saih8KrVj3f45ABKY%3D'
    # Load this file into a table, where each row is:
    # Tile ID, Xstart, Ystart, UZ, EPSG, MinLon, MaxLon, MinLat, MaxLat
    print('Reading HLS tile extents...')
    fp = temp_dir + '/hls_extents.csv'
    download_url(hls_tile_extents_url, fp)
    # s = requests.get(hls_tile_extents_url).content
    hls_tile_extents = pd.read_csv(fp, delimiter=r'\s+')
    print('Read HLS tile extents for {} tiles'.format(len(hls_tile_extents)))
    return hls_tile_extents
