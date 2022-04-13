"""Create an AzureML Workspace to train models in.

Based on https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=python

Example:
python 01-create-workspace.py \
    --name="my-cool-workspace" \
    --subscription-id="XXX" \
    --resource-group="ABC" \
    --region="eastus"
"""
import argparse
import os

from azureml.core import Workspace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help="Name for the workspace")
    parser.add_argument('--subscription-id', default = os.environ['WKSPC_SUBSCRIPTION_ID'], help="Azure Subscription ID")
    parser.add_argument('--resource-group', default = os.environ['CMPT_RESOURCE_GROUP'], help="Azure resource group")
    parser.add_argument('--region', default='eastus', help="What region to create the AzureML workspace in")
    args = parser.parse_args()

    ws = Workspace.create(name=args.name,
                          subscription_id=args.subscription_id,
                          resource_group=args.resource_group,
                          create_resource_group=False,
                          location=args.region)

    # write out the workspace details to a configuration file: .azureml/config.json
    ws.write_config(path='.azureml')
