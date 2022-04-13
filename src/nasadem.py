# Adapted from https://azure.microsoft.com/en-us/services/open-datasets/catalog/nasadem/
# Mostly-standard imports
import os
import tempfile
import urllib
import math

import numpy as np
import richdem as rd
import xarray as xr


# Storage locations are documented at http://aka.ms/ai4edata-nasadem
nasadem_account_name = 'nasadem'
nasadem_container_name = 'nasadem-nc'
nasadem_account_url = 'https://' + nasadem_account_name + '.blob.core.windows.net'
nasadem_blob_root = nasadem_account_url + '/' + nasadem_container_name + '/v001/'

# A full list of files is available at:
#
# https://nasademeuwest.blob.core.windows.net/nasadem-nc/v001/index/file_list.txt
nasadem_file_index_url = nasadem_blob_root + 'index/nasadem_file_list.txt'

nasadem_content_extension = '.nc'
nasadem_file_prefix = 'NASADEM_NC_'

# This will contain just the .nc files
nasadem_file_list = None

temp_dir = os.path.join(tempfile.gettempdir(),'nasadem')
os.makedirs(temp_dir,exist_ok=True)


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


def lat_lon_to_nasadem_tile(lat,lon):
    """
    Get the NASADEM file name for a specified latitude and longitude
    """

    # A tile name looks like:
    #
    # NASADEM_NUMNC_n00e016.nc
    #
    # The translation from lat/lon to that string is represented nicely at:
    #
    # https://dwtkns.com/srtm30m/

    # Force download of the file list
    get_nasadem_file_list()

    ns_token = 'n' if lat >=0 else 's'
    ew_token = 'e' if lon >=0 else 'w'

    lat_index = abs(math.floor(lat))
    lon_index = abs(math.floor(lon))

    lat_string = ns_token + '{:02d}'.format(lat_index)
    lon_string = ew_token + '{:03d}'.format(lon_index)

    filename = nasadem_file_prefix + lat_string + lon_string + \
        nasadem_content_extension

    if filename not in nasadem_file_list:
        print('Lat/lon {},{} not available'.format(lat,lon))
        filename = None

    return filename


def get_nasadem_file_list():
    """
    Retrieve the full list of NASADEM tiles
    """

    global nasadem_file_list
    if nasadem_file_list is None:
        nasadem_file = download_url(nasadem_file_index_url)
        with open(nasadem_file) as f:
            nasadem_file_list = f.readlines()
            nasadem_file_list = [f.strip() for f in nasadem_file_list]
            nasadem_file_list = [f for f in nasadem_file_list if \
                                 f.endswith(nasadem_content_extension)]
    return nasadem_file_list


def get_nasadem_tiles(lats, lons):
    return [
        nasadem_blob_root + lat_lon_to_nasadem_tile(lat, lon)
        for lat, lon in zip(lats, lons)
    ]


def get_nasadem_dataset(tile):
    url = nasadem_blob_root + tile
    pth = download_url(url, tile)
    ds = xr.open_dataset(pth).rename_vars({'NASADEM_HGT': 'elevation'}).drop_vars('crs')
    rd_arr = rd.rdarray(ds['elevation'].values, no_data=0)
    aspect = rd.TerrainAttribute(rd_arr, attrib='aspect')
    aspect[aspect < -9990] = 0.0
    # cos - range is -1 to 1 but we want 0 to 1
    aspect = (np.cos((aspect / 360) * 2 * math.pi) + 1) / 2
    # range is 0 to 90
    slp = rd.TerrainAttribute(rd_arr, attrib='slope_riserun')
    slp[slp < -9990] = 0.0
    # slope is 0 to 90, normal to [0-1]
    slp = slp / 90.0
    ds['slope'] = (ds.dims, slp)
    ds['aspect'] = (ds.dims, aspect)
    # normalize elevation to [0-1] (denali is 6190.5m in elevation, death valley is -86m)
    ds['elevation'] = (ds['elevation'] + 86.0) / (6190.5 - -86.0)
    return ds

