import argparse
import os

# import fiona before rasterio, rio-cogeo, and rioxarray cause they break something in fiona if imported first
import fiona

import numpy as np
import rioxarray as rioxr
import torch
import xarray as xr
import yaml

from affine import Affine
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from torch.nn import DataParallel

import model
from prediction_dataset import DatasetGenerator
from utils import get_dataloader
from utils import get_logger

logger = get_logger(__name__)

BANDS = ['class', 'basal_area', 'bio_acre', 'canopy_cvr']

# Sat settings
# LOG_INTERVAL = 100
# SAVE_INTERAL = 1000
# BATCH_SIZE = 320
# NUM_CLASSES = 5
# DATA_LOADER_WORKERS = 6

# HLS settings
LOG_INTERVAL = 100
SAVE_INTERAL = 3000
BATCH_SIZE = 6000
NUM_CLASSES = 5
DATA_LOADER_WORKERS = 12


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(model_path, csv_file, ds_generator, params, output_path, scales, checkpoint_path):
    # create necessary output directories
    make_dirs(output_path)

    # Load the trained model
    model_cls = getattr(model, params["model"])
    net = model_cls(
        num_classes=NUM_CLASSES if params['classify'] else 0,
        num_regressions=len(params['regression_vars']),
        naip_size=params.get('naip_size'),
        hls_size=params.get('hls_size')
    )
    net.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu' if not params['use_gpu'] else 'cuda'))
    )

    scales_tensor = torch.tensor(scales)
    if params['use_gpu']:
        net = DataParallel(net)
        net = net.to(device)
        scales_tensor = scales_tensor.to(device)
    net.eval()


    # predict for each tile
    for tile, attrs, ds in ds_generator:
        logger.info(f"Starting prediction for {tile.tile} {tile.year}")
        dl = get_dataloader(ds, None, {"batch_size": BATCH_SIZE}, num_workers=DATA_LOADER_WORKERS)
        output_tile = torch.zeros(3660, 3660, int(params['classify']) + len(params['regression_vars']))
        try:
            for i, ((xs, ys), inputs) in enumerate(dl):
                if params["use_gpu"]:
                    inputs = [inp.to(device) for inp in inputs]
                pred = net(*inputs)
                class_pred = pred[0].data.max(1, keepdim=True)[1]
                reg_pred = pred[1].data
                output_tile[ys, xs, :] = torch.cat([class_pred, reg_pred * scales_tensor], dim=1).cpu()
                if (i+1) % LOG_INTERVAL == 0:
                    logger.info(i+1)
                if (i+1) % SAVE_INTERAL == 0:
                    save_tile(output_tile, tile.tile, tile.year, attrs, scales, output_path)
            save_tile(output_tile, tile.tile, tile.year, attrs, scales, output_path)
            with open(checkpoint_path, 'a') as f:
                f.write(f"{tile.tile},{tile.year}\n")
        except Exception:
            logger.exception("uh oh")


def make_dirs(output_path):
    for band in BANDS:
        for year in range(2015, 2020):
            try:
                os.makedirs(f"{output_path}/{band}/{year}", exist_ok=True)
            except:
                pass


def _read_checkpoints(path):
    """Read checkpoints from file, or create file if it doesn't exist."""
    try:
        with open(path, 'r') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        logger.warning('No checkpoint file found, creating it at %s', path)
        with open(path, 'x') as f:
            pass
        return []


def save_tile(output_tile, curr_tile, curr_year, attrs, scales, output_path):
    """Save on-disk Cloud-optimized geotiffs (COG) one for each band
    """
    arr = np.transpose(output_tile.cpu().numpy(), (2, 0, 1))
    profile = dict(
        driver="GTiff",
        dtype="float32",
        count=1,
        height=3660,
        width=3660,
        crs=attrs['crs'],
        transform=Affine(*attrs['transform']),
    )
    for i in range(0, arr.shape[0]):
        band = np.expand_dims(arr[i, :, :], axis=0)
        with MemoryFile() as memfile:
            with memfile.open(**profile) as mem:
                mem.write(band)
                dst_profile = cog_profiles.get("deflate")
                cog_translate(
                    mem,
                    f"{output_path}/{BANDS[i]}/{curr_year}/{curr_tile}.tif",
                    dst_profile,
                    in_memory=True,
                    quiet=True,
                )
    logger.info(f"Completed {curr_tile} {curr_year}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--output-path',
        default='outputs',
        help='Location to save output tiles to'
    )
    parser.add_argument('--checkpoint-path', default='checkpoints.txt')
    parser.add_argument('--model-path')
    parser.add_argument('--tiles-path')
    parser.add_argument('--sources-path', help="Path to sources file, see src/configs/sources/template.yml for details")
    parser.add_argument('--params-path', help="Path to params file, see src/configs/template.yml for details")
    parser.add_argument('-s', '--scales', type=float, nargs='+', help='Scale factors for each regression variable. Printed by the training script.')
    args = parser.parse_args()

    with open(args.params_path) as f:
        params = yaml.load(f)
    with open(args.sources_path) as f:
        sources = yaml.load(f)
    params["use_gpu"] = params["use_gpu"] and torch.cuda.is_available()

    checkpoints = _read_checkpoints(args.checkpoint_path)
    ds_gen = DatasetGenerator(
        args.tiles_path,
        storage_account=sources.get('storage_account'),
        account_key=sources.get('account_key'),
        hls_path=sources.get('hls_tiles_path'),
        nasadem_path=sources.get('dem_path'),
        daymet_path=sources.get('daymet_tiles_path'),
        use_naip=sources.get('naip_path') is not None,
        naip_size=params.get('naip_size'),
        use_hls=sources.get('hls_path') is not None,
        hls_size=params.get('hls_size'),
        checkpoints=checkpoints,
    )
    predict(
        args.model_path,
        args.tiles_path,
        ds_gen,
        params,
        args.output_path,
        args.scales,
        args.checkpoint_path,
    )

    logger.info('Finished Predictions')
