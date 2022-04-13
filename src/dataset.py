import random

import fsspec
import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from torch.utils.data import IterableDataset

IMG_ROTATION_COUNT = list(range(4))


class FDIterableDataset(IterableDataset):
    def __init__(
        self,
        csv_file,
        hls_path=None,
        dem_path=None,
        daymet_path=None,
        naip_path=None,
        limit_to_naip=True,
        limit_to_hls=True,
        limit_to_dem=True,
        storage_account=None,
        account_key=None,
        forest_only=False,
        state_codes=[],
        ecocode_prefixes=[],
        classes=[],
        hls_size=32,
        naip_size=256,
        seed=1,
        regression_vars=[],
        outlier_percentile=None,
        normalization=None,
        extra_filters=False,
    ):
        """Dataset compatible with both Iterable and normal Dataset.

        Args:
            csv_file (str): Path to the csv file of all training samples
            hls_path (Optional[str]): Path to hls data or None to skip
            dem_path (Optional[str]): Path to dem data or None to skip
            daymet_path (Optional[str]): Path to daymet data or None to skip
            naip_path (Optional[str]): Path to naip data or None to skip
            limit_to_naip (bool): Whether to limit samples to those that have naip data
            limit_to_hls (bool): Whether to limit samples to those that have hls data
            limit_to_dem (bool): Whether to limit samples to those that have dem data
            storage_account (Optional[str]): If datasets are stored in blob storage - the name of the account they're in
            account_key (Optional[str]): If datasets are stored in blob storage - account key to access them
            forest_only (bool): Whether to eliminate "Non-forest" samples
            state_codes (List[str]): State codes to limit samples to
            ecocode_prefixes (List[str]): ECOCODE Prefixes to limit samples to
            hls_size (int): size to center crop each HLS chip to. Must be even and <=32
            naip_size (int): size to center crop each NAIP chip to. Must be even and <=256
            seed (int): Seed to use
            regression_vars (List[str]): List of columns to output as regression vars
            outlier_percentile (Optional[float]): If set remove any forested samples which has a regression variable above this percentile
            extra_filters (Optional[bool]): If set, remove any samples with non-zero canopy cover and zero AGB
        """
        self.naip_path = naip_path
        self.dem_path = dem_path
        self.hls_path = hls_path
        self.daymet_path = daymet_path
        self.limit_to_naip = limit_to_naip
        self.limit_to_hls = limit_to_hls
        self.limit_to_dem = limit_to_dem
        self.storage_account = storage_account
        self.account_key = account_key
        self.forest_only = forest_only
        self.hls_size = hls_size
        self.naip_size = naip_size
        self.pts = pd.read_csv(csv_file).set_index('INDEX')
        self.normalization = normalization
        self.regression_vars = regression_vars
        self.seed = seed
        self.extra_filters = extra_filters
        self._hls_ftype = '.npy'
        random.seed(self.seed)

        if outlier_percentile and len(self.regression_vars) > 0:
            forested = self.pts[self.pts['PLOT_STATUS'] == 'Forest']
            ols = [(forested[rv] >= forested[rv].quantile(outlier_percentile)) for rv in regression_vars]
            outliers = ols[0]
            for o in ols:
                outliers = (outliers | o)
            self.pts = self.pts.drop(forested[outliers].index)
        if self.limit_to_naip:
            self.pts = self.pts.drop(self.pts[self.pts.naip.isna()].index)
        if self.limit_to_hls:
            self.pts = self.pts.drop(self.pts[self.pts.tile.isna()].index)
        if self.limit_to_dem:
            self.pts = self.pts.drop(self.pts[self.pts.dem.isna()].index)

        if self.forest_only:
            self.pts = self.pts.drop(self.pts[self.pts.PLOT_STATUS!="Forest"].index)
        # Drop those that don't have a manually labeled FORESTED value of (0) and have a PLOT_STATUS != "Forest"
        # This throws out noisy Non-Forested data
        else:
            self.pts = self.pts.drop(self.pts[(self.pts.PLOT_STATUS != "Forest") & (self.pts.FORESTED != 0)].index)
        # Remove samples that have a canopy cover value but zero for other regression targets
        # Also remove measurements with no available canopy cover metric
        if self.extra_filters:
            self.pts = self.pts.drop(self.pts[(self.pts.CANOPY_CVR > 0) & (self.pts.BIO_ACRE == 0)].index)
            self.pts = self.pts.drop(self.pts[(self.pts.CANOPY_CVR == 0) & (self.pts.BIO_ACRE > 0)].index)
        if state_codes:
            self.pts = pd.concat([
                self.pts[self.pts.STATECD == code]
                for code in state_codes
            ])
        if classes:
            self.pts = pd.concat([
                self.pts[self.pts.CLASS == c]
                for c in classes
            ])
        if ecocode_prefixes:
            self.pts = pd.concat([
                self.pts[self.pts.ECOCODE.str.match(prefix)]
                for prefix in ecocode_prefixes
            ])

        self.regression_maxs = torch.tensor([
            self.pts[var].max() for var in self.regression_vars
        ]).float()
        self.regression_means = torch.tensor([
            self.pts[var].mean() for var in self.regression_vars
        ]).float()
        self.regression_stds = torch.tensor([
            self.pts[var].std() for var in self.regression_vars
        ]).float()


        if self.daymet_path:
            mean_path = _get_compatible_fsmap(
                self.daymet_path,
                "fia_40yr_monthly_means_norm.zarr",
                self.storage_account,
                self.account_key
            )
            std_path = _get_compatible_fsmap(
                self.daymet_path,
                "fia_40yr_monthly_std_devs_norm.zarr",
                self.storage_account,
                self.account_key
            )
            self.daymet_mean = xr.open_zarr(mean_path).compute()
            self.daymet_std = xr.open_zarr(std_path).compute()

        if self.dem_path:
            dem_path = _get_compatible_fsmap(
                self.dem_path,
                "nasadem-samples.zarr",
                self.storage_account,
                self.account_key
            )
            self.dem_data = xr.open_zarr(dem_path).compute()

        self.lst = [
            dict(idx=k, **v)
            for k, v in self.pts.to_dict('index').items()
        ]
        random.shuffle(self.lst)
        #Check to see if HLS samples are zarr or npy
        tstidx, tsttile = self.lst[0]['idx'], self.lst[0]['tile']
        full_path = _get_full_item_path(self.hls_path, f"{tstidx}-{tsttile}.npy")
        try:
            with fsspec.open(full_path) as f:
                testarr = np.load(f)
            self._hls_ftype = '.npy'
        except FileNotFoundError:
            self._hls_ftype = '.zarr'
    def get_regression_maxs(self):
        return self.regression_maxs

    def get_regression_means(self):
        return self.regression_means

    def get_regression_stds(self):
        return self.regression_stds

    def get_class_weights(self):
        """From https://github.com/scikit-learn/scikit-learn/issues/4324#issuecomment-76858134"""
        counts = torch.tensor(
            self.pts.groupby('CLASS').count().to_numpy()[:, 0]
        ).float()
        weights = counts.sum() / (counts.shape[0] * counts)
        return weights

    def stream_data(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # In this case we are not loading through a DataLoader with multiple workers
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # We only want to shuffle the order we traverse the files if we are the first worker (else, every worker will shuffle the files...)
        if worker_id == 0:
            random.shuffle(self.lst) # in place
        print(f"creating stream for {worker_id}")

        lst_len = len(self.lst)
        samples_per_worker = int(np.ceil(lst_len/num_workers))
        lower_bound = worker_id * samples_per_worker
        upper_bound = min(lst_len, (worker_id+1) * samples_per_worker)
        for idx in range(lower_bound, upper_bound):
            yield self[idx]

    def __getitem__(self, idx):
        pt = self.lst[idx]
        fia_idx = pt['idx']
        naip = None
        hls = None
        climate = None
        dem = None
        rots = random.choice(IMG_ROTATION_COUNT)
        if self.naip_path:
            full_path = _get_full_item_path(self.naip_path, f"{fia_idx}.tif")
            with rasterio.open(full_path) as img:
                arr = np.nan_to_num(img.read())
            # normalize to 0-1, and randomly rotate by 0, 90, 180, or 270 degrees
            arr = np.rot90(arr / 255, k=rots, axes=(1, 2)).copy()
            # center crop naip arr
            _, y, x = arr.shape
            startx, starty = x // 2 - self.naip_size // 2, y // 2 - self.naip_size // 2
            arr = arr[:, starty:starty+self.naip_size, startx:startx+self.naip_size]

            naip = torch.from_numpy(arr).float()
        if self.hls_path:
            full_path = _get_full_item_path(self.hls_path, f"{fia_idx}-{pt['tile']}{self._hls_ftype}")
            if self._hls_ftype == '.npy' :
                with fsspec.open(full_path) as f:
                    arr = np.load(f)
            else:
                arr = xarray.open_zarr(full_path).to_array().to_numpy()
            # normalize to 0-1, switch month and channel axis, and randomly rotate by 0, 90, 180, or 270 degrees
            arr = np.rot90(np.moveaxis(arr / 10000, 1, 0), k=rots, axes=(2, 3)).copy()
            # center crop hls arr
            _, _, y, x = arr.shape
            startx, starty = x // 2 - self.hls_size // 2, y // 2 - self.hls_size // 2
            arr = arr[:, :, starty:starty+self.hls_size, startx:startx+self.hls_size]

            hls = torch.from_numpy(arr).float()
        if self.daymet_path:
            pt_means = self.daymet_mean.sel({'idx': fia_idx}).to_array().values
            pt_std = self.daymet_std.sel({'idx': fia_idx}).to_array().values
            climate = torch.from_numpy(np.nan_to_num(np.stack([pt_means, pt_std]))).float()
        if self.dem_path:
            dem = torch.from_numpy(
                np.nan_to_num(self.dem_data.sel({'idx': fia_idx}).to_array().values)
            ).float()

        data = [
            data
            for data in [naip, hls, climate, dem]
            if data is not None
        ]

        if self.normalization == 'feature-scaling':
            regression_vars = torch.tensor([pt[var] for var in self.regression_vars]) / self.regression_maxs
        elif self.normalization == 'standard-score':
            regression_vars = (torch.tensor([pt[var] for var in self.regression_vars]) - self.regression_means) / self.regression_stds
        elif self.normalization is None:
            regression_vars = torch.tensor([pt[var] for var in self.regression_vars])

        class_var = torch.tensor(int(pt['CLASS']))  # Forest = 0, Non-Forest = 1
        return data, [class_var, regression_vars]

    def __iter__(self):
        return iter(self.stream_data())

    def __len__(self):
        return len(self.lst)


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
