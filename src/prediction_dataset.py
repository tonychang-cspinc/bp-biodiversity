import itertools

import fiona.transform
import fsspec
import pandas as pd
import torch
import xarray as xr
import numpy as np
from affine import Affine
from torch.utils.data import IterableDataset

from src.naip import NAIPTileIndex

from src.utils import get_logger

logger = get_logger(__name__)

# sort the bands alphanumerically because the bands in the training data are alphanumerically sorted as well
HLS_BANDS = sorted(['COASTAL_AEROSOL', 'BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR1', 'SWIR2'])


class DatasetGenerator:
    """Generates a single dataset for each row of the csv file - this ensures that each prediction dataset is only ever applying to one tile
    """
    def __init__(
        self,
        csv_file,
        hls_path=None,
        use_hls=False,
        use_naip=False,
        daymet_path=None,
        nasadem_path=None,
        storage_account=None,
        account_key=None,
        hls_size=4,
        naip_size=128,
        checkpoints=[],
    ):
        self.use_hls = use_hls
        self.use_naip = use_naip
        self.hls_path = hls_path
        self.daymet_path = daymet_path
        self.nasadem_path = nasadem_path
        self.storage_account = storage_account
        self.account_key = account_key
        self.tiles = pd.read_csv(csv_file)
        self.hls_size = hls_size
        self.naip_size = naip_size
        self.checkpoints = checkpoints

        self.naip_index = None
        if self.use_naip:
            self.naip_index = NAIPTileIndex()
    def __iter__(self):
        for t_index, t in self.tiles.iterrows():
            if f"{t.tile},{t.year}" in self.checkpoints:
                logger.info(f"Skipping prediction for {t.tile} {t.year}")
                continue
            hls_path = _get_compatible_fsmap(
                self.hls_path,
                f"{float(t.year)}/{t.tile}.zarr",
                self.storage_account,
                self.account_key
            )
            try:
                hls_tile = xr.open_zarr(hls_path)
            except Exception:
                logger.exception(f"Failed to read HLS tile {t.tile} {t.year}")
                logger.info(f'{self.storage_account}, {self.account_key}')
                break
            ds = PredictionDataset(
                t.tile,
                t.year,
                hls_path=self.hls_path,
                use_hls=self.use_hls,
                naip_index=self.naip_index,
                daymet_path=self.daymet_path,
                nasadem_path=self.nasadem_path,
                storage_account=self.storage_account,
                account_key=self.account_key,
                hls_size=self.hls_size,
                naip_size=self.naip_size
            )
            yield t, hls_tile.attrs, ds


class PredictionDataset(IterableDataset):
    def __init__(
        self,
        tile,
        year,
        hls_path=None,
        use_hls=False,
        naip_index=None,
        daymet_path=None,
        nasadem_path=None,
        storage_account=None,
        account_key=None,
        hls_size=4,
        naip_size=128
    ):
        """

        Args:
            hls_path (Optional[str]): Path to hls data
            use_hls (bool): Whether to include hls data or not
            use_dem (bool): Whether to include dem data or not
            naip_idx (Optional[NAIPTileIndex]): NAIP tile index to use or None to skip
            daymet_path (Optional[str]): Path to daymet data or None to skip
            storage_account (Optional[str]): If datasets are stored in blob storage - the name of the account they're in
            account_key (Optional[str]): If datasets are stored in blob storage - account key to access them
            hls_size (int): HLS chip size
            naip_size (int): NAIP chip size
        """
        self.tile = tile
        self.year = year
        self.hls_path = hls_path
        self.use_hls = use_hls
        self.naip_index = naip_index
        self.daymet_path = daymet_path
        self.nasadem_path = nasadem_path
        self.storage_account = storage_account
        self.account_key = account_key
        self.hls_size = hls_size
        self.naip_size = naip_size

        # load hls tile
        hls_path = _get_compatible_fsmap(
            self.hls_path,
            f"{float(self.year)}/{self.tile}.zarr",
            self.storage_account,
            self.account_key
        )
        try:
            self.hls_tile = xr.open_zarr(hls_path)
            logger.info(f"Streaming {self.tile} {self.year}")
        except Exception:
            logger.info(f"Failed to read HLS tile {self.tile} {self.year}")
            pass

        self.hls_crs = self.hls_tile.attrs['crs']
        self.hls_tfm = Affine(*self.hls_tile.attrs['transform'])

        # prep hls data as a tensor
        if self.use_hls:
            half_chip = self.hls_size // 2
            nparr = self.hls_tile[HLS_BANDS] \
                .to_array() \
                .fillna(0) \
                .pad({'x': (half_chip, half_chip), 'y': (half_chip, half_chip)}, mode='reflect') \
                .transpose('month', 'variable', 'y', 'x') \
                .data
            if nparr.shape[0] != 12:
                print('Data is missing months!')
                print('Filling Values with closest months....')
                colmonths = self.hls_tile.month.values
                badmonths = [x for x in np.arange(1,13) if x not in colmonths]
                outcube = np.zeros((12,len(HLS_BANDS),3660+self.hls_size,3660+self.hls_size))
                bmc = 0
                for m in np.arange(1,13):
                    if m in badmonths:
                        copyind = np.argmin(abs(colmonths-m))
                        bmc+=1
                    else:
                        copyind = m - 1 - bmc
                    outcube[m-1,:,:,:] = nparr[copyind,:,:,:]
                nparr = outcube
            self.hls_tensor = torch.from_numpy(nparr).float()

        if self.daymet_path:
            daymet_path = _get_compatible_fsmap(
                self.daymet_path,
                f"{self.tile}.zarr",
                self.storage_account,
                self.account_key
            )
            # self.daymet_tile = xr.open_zarr(daymet_path).compute()
            self.daymet_tensor = torch.from_numpy(
                xr.open_zarr(daymet_path)
                .to_array()
                .fillna(0)
                .transpose('variable','month', 'y', 'x')
                .data
            )
            shape = self.daymet_tensor.shape
            self.daymet_tensor = torch.reshape(self.daymet_tensor, (2, 5, 12, shape[-2], shape[-1]))
        if self.nasadem_path:
            nasadem_path = _get_compatible_fsmap(
                self.nasadem_path,
                f"{self.tile}.zarr",
                self.storage_account,
                self.account_key
            )
            # self.dem_tile = xr.open_zarr(nasadem_path).compute()
            self.nasadem_tensor = torch.from_numpy(
                xr.open_zarr(nasadem_path)
                .to_array()
                .fillna(0)
                .transpose('variable', 'y', 'x')
                .data
            )

    def stream_data(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # incorporate NAIP data
        if self.naip_index:
            naip_urls = self.naip_index.intersect_hls_tile(self.tile, int(self.year))
            i = 0
            try:
                for nm, naip_url in enumerate(naip_urls):
                    if nm % num_workers != worker_id:
                        continue
                    logger.info(f"NAIP subset: {naip_url}")
                    naip_tile = xr.open_rasterio(naip_url)[:3, :, :].compute()
                    naip_tile.attrs['nodatavals'] = (0, 0, 0, 0)

                    naip_tfm = Affine(*naip_tile.attrs['transform'])
                    naip_shape = naip_tile.shape
                    try:
                        xys = get_naip_subset(naip_tile, self.hls_tile)
                    except Exception:
                        logger.exception(f"Clip failed for: {naip_url}")
                        continue

                    # prep naip data as a tensor and put on GPU
                    naip_tensor = prepare_naip_tensor(naip_tile, self.naip_size)

                    for x, y in xys:
                        hls_input, naip_input, daymet_input, dem_input = None, None, None, None
                        ix, iy = xy_to_idx((x, y), self.hls_tfm)

                        if self.use_hls:
                            hls_input = self.hls_tensor[:, :, iy:iy+self.hls_size, ix:ix+self.hls_size]

                        if self.naip_index:
                            # transform hls coords into naip coords
                            nx, ny = fiona.transform.transform(self.hls_crs, naip_tile.attrs['crs'], [x], [y])
                            # transform naip coords into array index
                            nix, niy = xy_to_idx((nx[0], ny[0]), naip_tfm)
                            # sometimes nix, niy are beyond the bounds because hls is 30m vs naip 1m
                            nix, niy = min(max(nix, 0), naip_shape[2]), min(max(niy, 0), naip_shape[1])
                            naip_input = naip_tensor[:, niy:niy+self.naip_size, nix:nix+self.naip_size]

                        data = [
                            data
                            for data in [naip_input, hls_input, daymet_input, dem_input]
                            if data is not None
                        ]
                        yield (ix, iy), data
                        i += 1

                    # memory leaking, lets try to force cleanup
                    naip_tile.close()
                    del naip_tile
                    del naip_tensor
            except Exception:
                logger.exception(f"Yielding data for {self.tile} {self.year}")
        # HLS only
        else:
            i = 0
            for ix in range(3660):
                for iy in range(3660):
                    if i % num_workers != worker_id:
                        i += 1
                        continue
                    naip = None
                    hls = None
                    climate = None
                    dem = None

                    # x, y = self.hls_tfm * [ix, iy]

                    try:
                        hls = self.hls_tensor[:, :, iy:iy+self.hls_size, ix:ix+self.hls_size]
                        if self.nasadem_path:
                            dx, dy = int((ix/3660) * self.nasadem_tensor.shape[-1]), int((iy/3660) * self.nasadem_tensor.shape[-2])
                            dem = self.nasadem_tensor[:, dy, dx]
                        if self.daymet_path:
                            cx, cy = int((ix/3660) * self.daymet_tensor.shape[-1]), int((iy/3660) * self.daymet_tensor.shape[-2])
                            climate = self.daymet_tensor[:, :, :, cy, cx]
                        data = [
                            d
                            for d in [naip, hls, climate, dem]
                            if d is not None
                        ]
                        yield (ix, iy), data
                    except Exception as e:
                        logger.exception(f"Yielding data for {self.tile} {self.year}")
                    finally:
                        i += 1

    def __iter__(self):
        return iter(self.stream_data())


def _get_full_item_path(path, filename):
    """Given a path which can be a url or local path, construct the path to filename.

    Args:
        path (str): Either a local path or a Blob SAS URL
        filename (str): filename to open in the path
    """
    if "http" in path:
        [url, qs] = path.split('?')
        return f"{url}/{filename}?{qs}"
    else:
        return f"{path}/{filename}"


def _get_compatible_fsmap(path, filename, storage_account, account_key):
    full_path = f"{path}/{filename}"
    return fsspec.get_mapper(
        full_path,
        account_name=storage_account,
        account_key=account_key
    )


def get_naip_subset(naip_tile, hls_tile):
    """Return the HLS coordinates that overlap the naip tile."""
    nbs = naip_tile.rio.reproject(hls_tile.attrs['crs']).rio.bounds()
    hbs = hls_tile.rio.bounds()
    intersect_bounds = (
        max(nbs[0], hbs[0]),
        max(nbs[1], hbs[1]),
        min(nbs[2], hbs[2]),
        min(nbs[3], hbs[3])
    )
    hls_subset = hls_tile.rio.clip_box(*intersect_bounds)
    xys = itertools.product(list(hls_subset.x.data), list(hls_subset.y.data))

    return xys


def prepare_naip_tensor(naip_tile, naip_size):
    """Given a naip xr.DataArray, convert it into an input tensor and return tensor

    Returns: Tuple(tensor, shape, attrs)
    """
    half_chip = naip_size // 2
    naip_tensor = torch.from_numpy(
            (naip_tile.fillna(0) / 255)
            .pad({'x': (half_chip, half_chip), 'y': (half_chip, half_chip)}, mode='reflect')
            .data
    ).float()

    return naip_tensor


def xy_to_idx(xy, tfm):
    ix, iy = ~tfm * (xy[0], xy[1])
    ix, iy = int(ix), int(iy)
    return ix, iy
