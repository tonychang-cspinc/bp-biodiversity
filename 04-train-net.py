"""Train a model on AzureML.

Example:
python 04-train-net.py \
    --cluster="gpu-cluster" \
    --sources-path="config/sources/hls.yml" \
    --params-path="config/sources/hls/all.yml" \
"""
import argparse
import os

import yaml
from azureml.core import Dataset
from azureml.core import Environment
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
from azureml.core import Workspace
from azureml.core.runconfig import MpiConfiguration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources-path', help="Path to sources file, see configs/sources/template.yml for details")
    parser.add_argument('--params-path', help="Path to params file, see configs/template.yml for details")
    parser.add_argument('--gold-data-path', help="Path to gold data. If not supplied pulls from a default Dataset on AzureML")
    parser.add_argument('--gold-data-filename', help="Name of the gold data file, either locally or in the Azureml Dataset. Defaults to one stored on AzureML")
    parser.add_argument('-c', '--cluster', default='gpu-cluster', help="name of cluster to run on")
    # Not functioning correctly.
    parser.add_argument('-w', '--wait-for-completion', action='store_true', help="Script waits for AzureML to complete.")

    args = parser.parse_args()
    with open(args.params_path) as f:
        params = yaml.load(f)
    with open(args.sources_path) as f:
        sources = yaml.load(f)

    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name=params["model"])

    if not args.gold_data_path:
        dataset = Dataset.get_by_name(ws, name='gold_data')
        args.gold_data_path = dataset.as_named_input('gold').as_mount()
    if not args.gold_data_filename:
        args.gold_data_filename = 'fia_gold_data.csv'
    script_args = [
        "--sources-path", args.sources_path,
        "--params-path", args.params_path,
        "--gold-data-path", args.gold_data_path,
        "--gold-data-filename", args.gold_data_filename,
    ]

    config = ScriptRunConfig(
        source_directory='./src',
        script='train.py',
        compute_target=args.cluster if params["use_gpu"] else 'cpu-cluster',
        arguments=script_args,
        distributed_job_config=MpiConfiguration(node_count=params["node_count"])
    )

    env = Environment("torch1.6")
    env.docker.enabled = True 
    env.docker.base_image = f"{os.environ['ACR_REGISTRY']}.azurecr.io/{os.environ['ACR_IMAGENAME']}:{os.environ['ACR_IMAGETAG']}"
    env.python.user_managed_dependencies = True
    env.docker.base_image_registry.address = f"{os.environ['ACR_REGISTRY']}.azurecr.io"
    env.docker.base_image_registry.username = sources["username"]
    env.docker.base_image_registry.password = sources["password"]

    config.run_config.environment = env

    run = experiment.submit(config)

    # add tags
    for k, v in params.items():
        run.tag(k, str(v))

    aml_url = run.get_portal_url()
    print("Submitted to compute cluster.")

    # https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch
    # Not currently working as intended
    # if args.wait_for_completion:
    #     run.wait_for_completion(show_output=True)

    #     # Create a model folder in the current directory
    #     os.makedirs('./models', exist_ok=True)

    #     fold_count = params.get("folds") or 1
    #     for i in range(1, fold_count+1):
    #         name = f"{params['model']}_{i}.pt"
    #         run.register_model(model_name=name, model_path=f'outputs/{name}')
    #         # Download the model from run history
    #         run.download_file(name=f'outputs/{name}', output_file_path=f'./model/{name}')
