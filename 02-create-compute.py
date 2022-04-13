"""Create a computer cluster in AzureML.

Based on https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python

Example:
python 02-create-compute.py \
    --name="gpu-cluster" \
    --vm-size="STANDARD_NC6" \
    --max-nodes=6
"""
import argparse

from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help="Name for cluster")
    parser.add_argument('--vm-size', help="VM type to use in the cluster, ref: https://docs.microsoft.com/en-us/azure/virtual-machines/sizes")
    parser.add_argument('--max-nodes', type=int, help="Maximum number of nodes in the cluster")
    args = parser.parse_args()

    # This automatically looks for a directory .azureml
    ws = Workspace.from_config()

    # Verify that the cluster does not exist already
    try:
        cluster = ComputeTarget(workspace=ws, name=args.name)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size=args.vm_size,
                                                               idle_seconds_before_scaledown=300,
                                                               min_nodes=0,
                                                               max_nodes=args.max_nodes)
        cluster = ComputeTarget.create(ws, args.name, compute_config)

    cluster.wait_for_completion(show_output=True)
