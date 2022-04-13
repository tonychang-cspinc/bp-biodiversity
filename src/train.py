import argparse
import os
from pathlib import Path

import horovod.torch as hvd
import pandas as pd
import torch
import torch.optim as optim
import yaml
from azureml.core import Run
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import model
from dataset import FDIterableDataset
from loss import FDLoss
from test import test_model
from utils import get_dataloader
from utils import get_logger


REPORT_BATCH_INTERVAL = 500
NUM_CLASSES = 5
CLASS_NAMES = ["None", "Conifer", "Deciduous", "Mixed", "Dead"]

logger = get_logger(__name__)

run = Run.get_context()


def split_dataset(ds, folds, test_percent, seed):
    total_length = len(ds)
    test_length = int(len(ds) * test_percent)  # floor
    train_length = len(ds) - test_length
    fold_length = int(train_length / folds)
    extra = train_length % folds
    subset_lengths = []
    for f in range(folds):
        ssl = fold_length
        if extra > 0:
            ssl += 1
            extra -= 1
        subset_lengths.append(ssl)
    subset_lengths.append(test_length)
    split = random_split(ds, subset_lengths, torch.Generator().manual_seed(seed))
    return split[:-1], split[-1]


def holdout_split_dataset(ds, validation_percent, test_percent, seed):
    total_length = len(ds)
    test_length = int(len(ds) * test_percent)  # floor
    validation_length = int(len(ds) * validation_percent)  # floor
    lengths = [total_length-test_length-validation_length, validation_length, test_length]
    split = random_split(ds, lengths, torch.Generator().manual_seed(seed))
    for i in split:
        logger.info(f"{len(i)}")
    return split


def save_model(path, net, fold):
    model_name = f"{net.__class__.__name__}_{fold}.pt"
    filepath = Path(path) / model_name
    torch.save(net.state_dict(), filepath)
    model = run.upload_file(str(filepath), str(filepath))


def save_metrics(path, metrics, set_name, fold):
    metrics_name = f"{set_name}_metrics_{fold}.csv"
    filepath = Path(path) / metrics_name
    pd.DataFrame(metrics).to_csv(filepath)


def train_epoch(net, epoch, train_loader, train_sampler, optimizer, weights, params):
    logger.info(f"epoch: {epoch}")
    train_sampler.set_epoch(epoch)
    net.train()

    ##### Set up loss fn #####
    loss_fn = FDLoss(params['class_loss'], params['regression_loss'], weights)

    running_loss = 0.0
    for i, (inputs, responses) in enumerate(train_loader, 0):
        if params["use_gpu"]:
            inputs = [inp.cuda() for inp in inputs]
            responses = [resp.cuda() for resp in responses]

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(*inputs)

        loss = loss_fn(responses[0], outputs[0], responses[1], outputs[1])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % REPORT_BATCH_INTERVAL == REPORT_BATCH_INTERVAL-1:
            loss = running_loss / REPORT_BATCH_INTERVAL
            run.log('loss', loss)  # log loss metric to AML
            logger.info(f'epoch={epoch}, batch={i + 1:5}: loss {loss:.4f}')
            running_loss = 0.0


def train_fold(folds, validation_index, test_set, maxs, means, stds, weights, params):
    ##### Set up data #####

    # no cross-validation
    if len(folds) == 1:
        train_set = folds[0]
        validation_set = folds[0]
    else:
        train_set = ConcatDataset(folds[:validation_index] + folds[validation_index+1:])
        validation_set = folds[validation_index]

    train_sampler = DistributedSampler(train_set, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = get_dataloader(train_set, train_sampler, params)

    ##### Set up net #####
    model_class = getattr(model, params["model"])
    net = model_class(
        num_classes=NUM_CLASSES if params['classify'] else 0,
        num_regressions=len(params['regression_vars']),
        hls_size=params.get('hls_size', 32),
        naip_size=params.get('naip_size', 256),
    )
    hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    if params["use_gpu"]:
        logger.info("Using GPU")
        net.cuda()

    ##### Set up optimizer #####
    optimizer = optim.Adam(
        net.parameters(),
        lr=params["learning_rate"] * hvd.size()
    )
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=net.named_parameters(),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params.get("learning_schedule_epochs", 20),
        gamma=params.get("learning_schedule_gamma", 1.0)
    )

    ##### Set up list for tracking metrics per epoch #####
    train_metrics = []
    valid_metrics = []

    ##### Train the network #####
    logger.info("Starting training")
    for epoch in range(1, params["epochs"]+1):
        train_epoch(
            net, epoch, train_loader, train_sampler, optimizer, weights, params
        )
        pred_output_dir = 'outputs' if epoch==params['epochs'] else None
        train_metrics.append(
            test_model(net, train_set, "train", maxs, means, stds, weights, params, pred_output_dir=pred_output_dir)
        )
        valid_metrics.append(
            test_model(net, validation_set, "validation", maxs, means, stds, weights, params, pred_output_dir=pred_output_dir)
        )

        scheduler.step()
    return net, train_metrics, valid_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--output-path',
        default='outputs',
        help='Location to save output models and data to'
    )
    parser.add_argument('--gold-data-path')
    parser.add_argument('--gold-data-filename')
    parser.add_argument('--sources-path', help="Path to sources file, see src/configs/sources/template.yml for details")
    parser.add_argument('--params-path', help="Path to params file, see src/configs/template.yml for details")
    args = parser.parse_args()

    with open(args.params_path) as f:
        params = yaml.load(f)
    with open(args.sources_path) as f:
        sources = yaml.load(f)

    params["use_gpu"] = params["use_gpu"] and torch.cuda.is_available()

    hvd.init()
    torch.manual_seed(params["seed"])
    logger.info(f"hvd local rank {hvd.local_rank()}")
    logger.info(f"hvd rank {hvd.rank()}")
    logger.info(f"hvd size {hvd.size()}")

    if params["use_gpu"]:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(params['seed'])

    
    dataset = FDIterableDataset(
        f"{args.gold_data_path}/{args.gold_data_filename}",
        limit_to_naip=False,
        limit_to_hls=True,
        limit_to_dem = False,
        naip_path=sources.get("naip_path"),
        hls_path=sources.get("hls_path"),
        dem_path=sources.get("dem_path"),
        daymet_path=sources.get("daymet_path"),
        account_key=sources.get("account_key"),
        storage_account=sources.get("storage_account"),
        forest_only=params["forest_only"],
        seed=params["seed"],
        ecocode_prefixes=params.get('ecocode_prefixes'),
        state_codes=params.get('state_codes'),
        classes=params.get('classes'),
        hls_size=params.get('hls_size', 32),
        naip_size=params.get('naip_size', 256),
        regression_vars=params.get('regression_vars'),
        normalization=params.get('normalization'),
        outlier_percentile=None,
        extra_filters=params.get("extra_filters")
    )
    logger.info(f"Dataset size: {len(dataset)}")
    maxs = dataset.get_regression_maxs()
    means = dataset.get_regression_means()
    stds = dataset.get_regression_stds()
    weights = dataset.get_class_weights()
    if params["use_gpu"]:
        weights = weights.cuda()
    logger.info(f"Dataset maxs {maxs}")
    logger.info(f"Dataset means {means}")
    logger.info(f"Dataset stds {stds}")

    # split dataset into train/validation and test sets
    if params["holdout_percent"]:
        subsets = holdout_split_dataset(dataset, params["holdout_percent"], params["test_percent"], params["seed"])
        net, train_metrics, valid_metrics = train_fold(subsets[:2], 1, subsets[2], maxs, means, stds, weights, params)
        if hvd.rank() == 0:
            save_metrics(args.output_path, train_metrics, "train", 1)
            save_metrics(args.output_path, valid_metrics, "validation", 1)
            save_model(args.output_path, net, 1)
    else:
        folds, test = split_dataset(dataset, params["folds"], params["test_percent"], params["seed"])
        for i in range(params["folds"]):
            net, train_metrics, valid_metrics = train_fold(folds, i, test, maxs, means, stds, weights, params)
            if hvd.rank() == 0:
                save_metrics(args.output_path, train_metrics, "train", i+1)
                save_metrics(args.output_path, valid_metrics, "validation", i+1)
                save_model(args.output_path, net, i+1)

    logger.info('Finished Training')
