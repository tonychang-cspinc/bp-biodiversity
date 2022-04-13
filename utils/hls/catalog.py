"""Tools for creating catalogs of HLS scenes and bands for bboxes/points

Example usage:
lookup = HLSTileLookup()
fia_df = fia_csv_to_data_catalog_input('./fia_10.csv')
bbox = [-124.98046874999999, 24.367113562651262, -66.70898437499999, 49.49667452747045]
years = [2019]
bands = [HLSBand.NIR_NARROW, HLSBand.QA]
point_catalog = HLSCatalog.from_point_pandas(df=fia_df, bands=[HLSBand.NIR_NARROW, HLSBand.QA], tile_lookup=lookup)
bbox_catalog = HLSCatalog.from_bbox(bbox, years, bands, sentinel_bands, lookup)
"""
import requests
import re
import io
from datetime import datetime
from functools import lru_cache
import concurrent.futures

import geopandas as gpd
import numpy as np
import pandas as pd
import rtree
import xarray as xr
from azure.storage.blob import ContainerClient
from shapely.geometry import box
from tqdm import tqdm
import rasterio
from rasterio.windows import Window

from enum import Enum

BAND_TO_L30 = {
    1: "01",
    2: "02",
    3: "03",
    4: "04",
    5: "05",
    6: "06",
    7: "07",
    8: "08",
    11: "11",
}

BAND_TO_S30 = {
    1: "01",
    2: "02",
    3: "03",
    4: "04",
    5: "09",
    6: "10",
    7: "11",
    8: "13",
    11: "14",
}


class HLSBand(Enum):
    """Enum for HLS band names.

    From https://azure.microsoft.com/en-us/services/open-datasets/catalog/hls/
    Usage:
        x = HLSBand.BLUE
        x.to_sensor_band_name('L') -->
    """

    COASTAL_AEROSOL = 1
    BLUE = 2
    GREEN = 3
    RED = 4
    NIR_NARROW = 5
    SWIR1 = 6
    SWIR2 = 7
    CIRRUS = 8
    QA = 11

    def to_sensor_band_name(self, sensor):
        if sensor == 'L':
            return BAND_TO_L30[self.value]
        else:
            return BAND_TO_S30[self.value]

class HLSTileLookup:
    """Wrapper around an rtree for finding HLS tile ids."""

    def __init__(self):
        self.tile_extents = self._get_extents()
        self.tile_gpd = self._get_boxes()
        self.tree_idx = rtree.index.Index()
        self.idx_to_id = {}
        self.idx_to_all = {}
        for i_row, row in self.tile_extents.iterrows():
            self.tree_idx.insert(i_row, (row.MinLon, row.MinLat, row.MaxLon, row.MaxLat))
            self.idx_to_id[i_row] = row.TilID

    def _get_extents(self):
        hls_tile_extents_url = 'https://ai4edatasetspublicassets.blob.core.windows.net/assets/S2_TilingSystem2-1.txt?st=2019-08-23T03%3A25%3A57Z&se=2028-08-24T03%3A25%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=KHNZHIJuVG2KqwpnlsJ8truIT5saih8KrVj3f45ABKY%3D'
        # Load this file into a table, where each row is:
        # Tile ID, Xstart, Ystart, UZ, EPSG, MinLon, MaxLon, MinLat, MaxLat
        print('Reading tile extents...')
        s = requests.get(hls_tile_extents_url).content
        hls_tile_extents = pd.read_csv(io.StringIO(s.decode('utf-8')),delimiter=r'\s+')
        print('Read tile extents for {} tiles'.format(len(hls_tile_extents)))
        return hls_tile_extents

    def _get_boxes(self):
        return gpd.GeoDataFrame(
            self.tile_extents,
            crs="EPSG:4326",
            geometry=[
                box(r['MinLon'], r['MinLat'], r['MaxLon'], r['MaxLat'])
                for _, r in self.tile_extents.iterrows()
            ]
        )

    def get_point_hls_tile_ids(self, lat, lon):
        results = list(self.tree_idx.intersection((lon, lat, lon, lat)))
        return set([self.idx_to_id[r] for r in results])

    def get_bbox_hls_tile_ids(self, left, bottom, right, top):
        return set(
            self.idx_to_id[i]
            for i in self.tree_idx.intersection((left, bottom, right, top))
        )

    def get_geometry_hls_tile_ids(self, geometry):
        tiles_in_aoi = gpd.overlay(geometry, self.tile_gpd, how='intersection')
        return set(tiles_in_aoi['TilID'])

    def get_hls_tile_info(self, tile_ids):
        return self.tile_extents[self.tile_extents['TilID'].isin(tile_ids)]

    def get_point_hls_tile_info(self, lat, lon):
        tile_ids = self.get_point_hls_tile_ids(lat, lon)
        return self.get_hls_tile_info(tile_ids)

    def get_bbox_hls_tile_info(self, left, bottom, right, top):
        tile_ids = self.get_bbox_hls_tile_ids(left, bottom, right, top)
        return self.get_hls_tile_info(tile_ids)

    def get_geometry_hls_tile_info(self, geometry):
        tile_ids = self.get_geometry_hls_tile_ids(geometry)
        return self.get_hls_tile_info(tile_ids)


class HLSCatalog:
    """Wrapper around an xarray dataset.

    Contains `self.xr_ds` which is an xarray dataset with at least:
        variables of: tile (string HLS tile id), scene, sensor, dt (datetime)
        attrs: desired bands

    Includes utility functions for constructing catalog from a pandas dataframe of training points or a bbox.
    Includes utility functions for reading/writing to zarr to persist catalog once its created.
    """

    def __init__(self, xr_ds, validate=False):
        self.xr_ds = xr_ds
        if validate:
            self._validate_catalog()
    def to_zarr(self, path):
        self.xr_ds.attrs['bands'] = [
            band.value
            for band in self.xr_ds.attrs['bands']
        ]
        self.xr_ds.to_zarr(path, mode='w')
        self.xr_ds.attrs['bands'] = [
            HLSBand(value)
            for value in self.xr_ds.attrs['bands']
        ]
    def drop_scenes(self,bad_urls):
        badtiles=np.unique([x.split('.')[-5] for x in bad_urls])
        badscenes=['/'.join(x.split('/')[-2:])[:-7] for x in bad_urls]
        indcopy=self.xr_ds.index.copy()
        

        self.xr_ds['index'] = np.arange(0,len(indcopy))
        badinds = np.where(self.xr_ds.data_vars['scene'].isin(badscenes))[0]
        if len(badinds) == 0:
            print('Given urls not found in catalog.')
        else:
            updatedinds = np.delete(indcopy,badinds)
            self.xr_ds=self.xr_ds.drop_sel({'index':badinds})
            self.xr_ds['index']=updatedinds
            nsize=len(indcopy)-len(badinds)
            print(f'Dropped {len(badinds)} scenes from the catalog. The new size is {nsize}.')
        
    @classmethod
    def from_zarr(cls, path):
        catalog = cls(xr.open_zarr(path)) 
        catalog.xr_ds.attrs['bands'] = [
            HLSBand(value)
            for value in catalog.xr_ds.attrs['bands']
        ]
        return catalog

    @classmethod
    def from_point_pandas(cls, df, bands=[], tile_lookup=None, include_scenes=True):
        """
        Args:
            df (pandas.DataFrame): Dataframe with at least the following columns: lat, lon, year
            bands list(HLSBand): list of bands to fetch
        """
        lookup = tile_lookup if tile_lookup else HLSTileLookup()
        df = df
        df['tile'] = df.apply(lambda row: lookup.get_point_hls_tile_ids(row.lat, row.lon), axis=1)
        df = df.explode('tile')
        if include_scenes:
            df = _tiles_to_scenes(df)
        # create xr_dataset
        xr_ds = xr.Dataset.from_dataframe(df)
        xr_ds.attrs['bands'] = bands
        return cls(xr_ds)

    @classmethod
    def from_bbox(cls, bbox, years, bands=[], tile_lookup=None, include_scenes=True):
        lookup = tile_lookup if tile_lookup else HLSTileLookup()
        tiles = list(lookup.get_bbox_hls_tile_ids(*bbox))
        df = pd.DataFrame(tiles).rename({0: 'tile'}, axis=1)
        df['years'] = [years] * len(tiles)
        df = df.explode('years').rename({'years': 'year'}, axis=1)
        if include_scenes:
            df = _tiles_to_scenes(df)
        # create xr_dataset
        xr_ds = xr.Dataset.from_dataframe(df)
        xr_ds.attrs['bands'] = bands
        return cls(xr_ds)

    @classmethod
    def from_geom(cls, geom, years, bands=[], tile_lookup=None, include_scenes=True):
        lookup = tile_lookup if tile_lookup else HLSTileLookup()
        tiles = list(lookup.get_geometry_hls_tile_ids(geom))
        df = pd.DataFrame(tiles).rename({0: 'tile'}, axis=1)
        df['years'] = [years] * len(tiles)
        df = df.explode('years').rename({'years': 'year'}, axis=1)
        if include_scenes:
            df = _tiles_to_scenes(df)
        # create xr_dataset
        xr_ds = xr.Dataset.from_dataframe(df)
        xr_ds.attrs['bands'] = bands
        return cls(xr_ds)
    @classmethod
    def from_tilesdf(cls, tilesdf, bands=[], tile_lookup=None, include_scenes=True):
        if include_scenes:
            df = _tiles_to_scenes(tilesdf)
        # create xr_dataset
        xr_ds = xr.Dataset.from_dataframe(df)
        xr_ds.attrs['bands'] = bands
        return cls(xr_ds)

    def _validate_catalog(self):
        sclist = list(self.xr_ds.data_vars['scene'].values.astype(str))
        master_badurls =[]
        print('Validating scene data. This will take some time...')
        def check_scene_validity(sc):
            sens = 'L' if 'L30' in sc else 'S'
            badurls=[]
            for url in scene_to_urls(sc, sens, self.xr_ds.attrs['bands']):
                try:
                    with rasterio.open(url) as f:
                        trans = f.transform.to_gdal()
                        if np.sum(trans) == 2:
                            raise rasterio.errors.RasterioIOError('Band is not georeferenced.')
                        topcheck = f.read(1, window=Window(0, 0, f.width, 10))
                        botcheck = f.read(1, window=Window(0, f.height-10, f.width, 10))
                except rasterio.errors.RasterioIOError as e:
                    badurls.append(url)
                    continue
            return badurls
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for broken_subset in tqdm(executor.map(check_scene_validity, sclist), total=len(sclist)):
                master_badurls.extend(broken_subset)
        self.drop_scenes(master_badurls)
        
def fia_csv_to_data_catalog_input(path):
    df = pd.read_csv(path)
    df.rename({"LAT": "lat", "LON": "lon"}, inplace=True, axis=1)
    df['INVYR'] = df['INVYR'].astype(int)
    df['years'] = df.apply(lambda x: np.arange(x.INVYR, x.INVYR+1), axis=1) # change to correct range
    return df.explode('years').rename({"years": "year"}, axis=1)


def _list_available_tiles(prefix, band="01"):
    hls_container_name = 'hls'
    hls_account_name = 'hlssa'
    hls_account_url = 'https://' + hls_account_name + '.blob.core.windows.net/'
    hls_blob_root = hls_account_url + hls_container_name
    hls_container_client = ContainerClient(
        account_url=hls_account_url,
        container_name=hls_container_name,
        credential=None
    )
    files = []
    generator = hls_container_client.list_blobs(name_starts_with=prefix)
    for blob in generator:
        if blob.name.endswith(f'{band}.tif'):
                        files.append(blob.name)
    return files



@lru_cache(maxsize=10000)
def _list_scenes(tile, year, folder, product):
    prefix = f'{folder}/HLS.{product}.T{tile}.{year}'
    urls = _list_available_tiles(prefix)
    data = []
    for url in urls:
        data.append(url.split('_')[0])
    return data


def _tiles_to_scenes(df):
    # join landsat and sentinel scenes
    tiles, years = df['tile'].values, df['year'].values
    N = len(tiles)
    landsat, sentinel = [],[]
    print('Searching for matching Landsat scenes...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for urllist in tqdm(executor.map(lambda tinfo: _list_scenes(*tinfo), zip(tiles, years, np.full(N, 'L30'), np.full(N,'L30'))),total=N):
            landsat.append(urllist)
    lseries = pd.Series(landsat)
    print('Searching for matching Sentinel scenes...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for urllist in tqdm(executor.map(lambda tinfo: _list_scenes(*tinfo),zip(tiles, years, np.full(N, 'S30'), np.full(N,'S30'))),total=N):
            sentinel.append(urllist)
    sseries = pd.Series(sentinel)
    df['scenes'] = lseries + sseries
    # filter out rows w/ empty scenes
    df = df[df.scenes.astype(bool)]
    # explode list of scenes into one row per scene
    df = df.explode('scenes').rename({'scenes': 'scene'}, axis=1)
    df['sensor'] = df.apply(lambda row: 'L' if 'L30' in row.scene else 'S', axis=1)
    # get datetime for scene
    df['dt'] = df.apply(lambda row: _scene_to_datetime(row.scene), axis=1)
    return df

def scene_to_urls(scene, sensor, bands):
    """Take a scene id and a list of HLSBand enums and constructs an Azure blob url for each band.

    Each result is loadable by xr.open_rasterio(url)

    Args:
        scene (str): String that looks something like S309/HLS.S30.T10TET.2019001.v1.4
        sensor (str): Either 'L' (Landsat) or 'S' (Sentinel-2)
        bands (Iterable[HLSBand]): List of bands to get urls for

    Returns:
        List[str]: List of urls, one for each band in the same order as the bands input
    """
    return [f"https://hlssa.blob.core.windows.net/hls/{scene}_{band.to_sensor_band_name(sensor)}.tif" for band in bands]


def _scene_to_datetime(scene):
    groups = re.search(r'.*(?P<yearday>\d{7}).*', scene)
    return datetime.strptime(groups['yearday'], '%Y%j')
