import horovod.torch as hvd
import pandas as pd
import torch
from azureml.core import Run
from torch.utils.data.distributed import DistributedSampler

from classify_metrics import ClassMetrics
from loss import FDLoss
from r2_score import R2Score
from rmse import RMSE
from utils import get_dataloader
from utils import get_logger

NUM_CLASSES = 5
CLASS_NAMES = ["None", "Conifer", "Deciduous", "Mixed", "Dead"]

logger = get_logger(__name__)

run = Run.get_context()

AZUREML_METRICS = {
    "train_loss",
    "train_accuracy",
    "train_precision_None",
    "train_recall_None",
    "train_f1_None",
    "train_precision_Conifer",
    "train_recall_Conifer",
    "train_f1_Conifer",
    "train_precision_Deciduous",
    "train_recall_Deciduous",
    "train_f1_Deciduous",
    "train_precision_Mixed",
    "train_recall_Mixed",
    "train_f1_Mixed",
    "train_precision_Dead",
    "train_recall_Dead",
    "train_f1_Dead",
    "train_r2_BASAL_AREA",
    "train_rmse_BASAL_AREA",
    "train_r2_BIO_ACRE",
    "train_rmse_BIO_ACRE",
    "train_r2_CANOPY_CVR",
    "train_rmse_CANOPY_CVR",
    "validation_loss",
    "validation_accuracy",
    "validation_precision_None",
    "validation_recall_None",
    "validation_f1_None",
    "validation_precision_Conifer",
    "validation_recall_Conifer",
    "validation_f1_Conifer",
    "validation_precision_Deciduous",
    "validation_recall_Deciduous",
    "validation_f1_Deciduous",
    "validation_precision_Mixed",
    "validation_recall_Mixed",
    "validation_f1_Mixed",
    "validation_precision_Dead",
    "validation_recall_Dead",
    "validation_f1_Dead",
    "validation_r2_BASAL_AREA",
    "validation_rmse_BASAL_AREA",
    "validation_r2_BIO_ACRE",
    "validation_rmse_BIO_ACRE",
    "validation_r2_CANOPY_CVR",
    "validation_rmse_CANOPY_CVR",
}


def log_metrics(dct):
    logger.info(dct)
    for k in AZUREML_METRICS:
        if dct.get(k):
            run.log(k, dct[k])


def metric_average(tensor, name):
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor


def test_model(net, test_set, dataset_name, maxs, means, stds, weights, params, pred_output_dir=False):
    logger.info(f"Starting model test on {dataset_name}")
    net.eval()
    if params["use_gpu"]:
        maxs = maxs.cuda()
        means = means.cuda()
        stds = stds.cuda()
    test_sampler = DistributedSampler(test_set, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = get_dataloader(test_set, test_sampler, params)

    loss_fn = FDLoss(params['class_loss'], params['regression_loss'], weights)

    test_loss = torch.zeros(1)
    test_accuracy = torch.zeros(1)
    r2 = R2Score(len(params['regression_vars']), NUM_CLASSES)
    rmse = RMSE(
        len(params['regression_vars']),
        NUM_CLASSES,
        maxs if params.get('normalization') == 'feature-scaling' else None,
        means if params.get('normalization') == 'standard-score' else None,
        stds if params.get('normalization') == 'standard-score' else None
    )
    cm = ClassMetrics(NUM_CLASSES)

    if pred_output_dir:
        results = torch.zeros(0, (len(params['regression_vars'])+int(params['classify']))*2)

    for i, data in enumerate(test_loader):
        inputs, responses = data

        if pred_output_dir:
            inp_data = torch.zeros(responses[0].shape[0], 0)
            if params['classify']:
                inp_data = torch.cat([inp_data, responses[0].unsqueeze(1)], dim=1)
            if params['regression_vars']:
                inp_data = torch.cat([inp_data, responses[1]], dim=1)

        if params["use_gpu"]:
            inputs = [inp.cuda() for inp in inputs]
            responses = [resp.cuda() for resp in responses]
        outputs = net(*inputs)

        loss = loss_fn(responses[0], outputs[0], responses[1], outputs[1])
        test_loss += loss.item()

        if params['classify']:
            class_pred = outputs[0].data.max(1, keepdim=True)[1]
            test_accuracy += class_pred.eq(responses[0].view_as(class_pred)).cpu().float().sum()
            cm.update(outputs[0].data, responses[0])
        if params['regression_vars']:
            rmse.update(outputs[1].data, responses[1], responses[0])
            r2.update(outputs[1].data, responses[1], responses[0])

        if pred_output_dir:
            out_data = torch.zeros(responses[0].shape[0], 0)
            if params['classify']:
                out_data = torch.cat([out_data, torch.argmax(outputs[0].data, dim=1).unsqueeze(1).cpu()], dim=1)
            if params['regression_vars']:
                out_data = torch.cat([out_data, outputs[1].data.cpu()], dim=1)
            new_data = torch.cat([inp_data, out_data], dim=1)
            results = torch.cat([results, new_data], dim=0)

    if pred_output_dir:
        cols = ['class'] + params['regression_vars']
        cols = cols + [f'pred_{col}' for col in cols]
        all_results = hvd.allgather(results, name="results")
        if hvd.rank() == 0:
            pd.DataFrame(
                all_results.numpy()
            ).rename(
                {i: c for i, c in enumerate(cols)}, axis=1
            ).to_csv(f'{pred_output_dir}/{dataset_name}_predictions.csv', index=False)

    test_loss /= (i+1)  # divide by number of batches as we summed the average per batch
    test_loss = metric_average(test_loss, 'avg_loss')

    if params['classify']:
        test_accuracy /= len(test_sampler)
        test_accuracy = metric_average(test_accuracy, 'avg_accuracy') * 100.0
        test_precision, test_recall, test_f1 = cm.compute()
    if params['regression_vars']:
        test_overall_rmse, test_class_rmse = rmse.compute()
        test_overall_r2, test_class_r2 = r2.compute()
    if hvd.rank() == 0:
        metrics_dict = {}
        metrics_dict[f'{dataset_name}_loss'] = test_loss.item()
        if params['classify']:
            metrics_dict[f'{dataset_name}_accuracy'] = test_accuracy.item()
            for i in range(NUM_CLASSES):
                cname = CLASS_NAMES[i]
                metrics_dict[f'{dataset_name}_precision_{cname}'] = test_precision[i].item()
                metrics_dict[f'{dataset_name}_recall_{cname}'] = test_recall[i].item()
                metrics_dict[f'{dataset_name}_f1_{cname}'] = test_f1[i].item()
        if params['regression_vars']:
            for i, name in enumerate(params['regression_vars']):
                metrics_dict[f"{dataset_name}_r2_{name}"] = test_overall_r2[i].item()
                metrics_dict[f"{dataset_name}_rmse_{name}"] = test_overall_rmse[i].item()
            for i, name in enumerate(params['regression_vars']):
                for j, cls_name in enumerate(CLASS_NAMES):
                    metrics_dict[f"{dataset_name}_{cls_name}_r2_{name}"] = test_class_r2[j, i].item()
                    metrics_dict[f"{dataset_name}_{cls_name}_rmse_{name}"] = test_class_rmse[j, i].item()

        log_metrics(metrics_dict)
        return metrics_dict
