import logging
import os
import sys

import torch

PREFETCH_FACTOR = 10


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    shandler = logging.StreamHandler(sys.stdout)
    shandler.setFormatter(formatter)
    logger.addHandler(shandler)
    return logger


def get_dataloader(dataset, sampler, params, num_workers=os.cpu_count()-1):
    # prefetch_factor only available in pytorch 1.7 and later
    if torch.__version__ >= '1.7.0':
        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=params["batch_size"],
            prefetch_factor=PREFETCH_FACTOR,
            num_workers=num_workers,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=params["batch_size"],
            num_workers=num_workers,
        )
    return loader
