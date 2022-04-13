"""Generate the gold data csv used for input to the training script.

Before running you must create a FileDataset in AzureML named `gold_data` that the resulting data from the script will be uploaded to.

Example: 
python 03-create-gold-data-csv.py \
    --plot-csv=fia_no_pltcn.csv \
    --response-csv=fia_ytrain.csv \
    --output-csv=modeling_data.csv \
    --account-key=XXXX \
    --non-forest-csv=labeled-non-forest.csv
"""
import argparse
import os

import fsspec
import pandas as pd
import xarray as xr
from azureml.core import Dataset, Workspace, Datastore
from azureml.data.datapath import DataPath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-csv', help="CSV of plot sample data. Columns: INDEX, INVYR (inventory year), STATECD (FIPS State code), LAT, LON")
    parser.add_argument('--response-csv', help="CSV of response variables and Bailey's ecoregions for each index. Columns: INDEX,TPA,BAA,CARB_ACRE,nStems,BIO_ACRE,PLOT_STATUS,INVYR,STATECD,CLASS,CANOPY_CVR,BASAL_AREA,ECO_US_ID,ECOCODE,DOMAIN,DIVISION,PROVINCE,SECTION")
    parser.add_argument('--non-forest-csv', help="CSV of 'non-forested' training points that truly are non-forested (have no trees). two columns: INDEX, FORESTED (always 1 or 0)")
    parser.add_argument('--output-csv', default = 'fia_gold_data.csv', help="Filename for the output csv")
    parser.add_argument('--account-key', default = os.environ["STRG_ACCOUNT_KEY"] help="Azure storage account key." )
    parser.add_argument('--account-name', default = os.environ["STRG_ACCOUNT_NAME"], help="Azure storage account name." )
    parser.add_argument('--blob-container', default = os.environ["BLOB_CONTAINER"]  help="Azure storage blob container name to store created csv.")
    parser.add_argument('--resource-group', , default = os.environ["STRG_RESOURCE_GROUP"], help="Azure resource group for new dataset with the single created csv." )
    parser.add_argument('--strg-subscription-id',default = None, help='Subscription ID for storage, if different from workspace.')
    parser.add_argument('--blob-datastore-name',default='fia',help='Name for newly created datastore.')
    parser.add_argument('--only-labeled-samples', action='store_true')
    parser.add_argument('--all-vars', default=1, type=int, help="Set to 0 to cancel parsing available naip and nasadem tiles")

    args = parser.parse_args()
    blob_datastore_name=args.blob_datastore_name

    ws = Workspace.from_config()
    if args.strg_subscription_id is None:
        args.strg_subscription_id=ws.get_details()['id'].split('/')[2]

    blob_datastore = Datastore.register_azure_blob_container(workspace=ws, 
                                                            datastore_name=blob_datastore_name, 
                                                            container_name=args.blob_container, 
                                                            account_name=args.account_name,
                                                            account_key=args.account_key,
                                                            resource_group=args.resource_group,
                                                            subscription_id=args.strg_subscription_id,
                                                            grant_workspace_access=True)

    
    pts = pd.read_csv(args.plot_csv).set_index('INDEX')
    response_vars = pd.read_csv(args.response_csv).set_index('INDEX').drop(columns=['STATECD', 'INVYR'])
    fs = fsspec.filesystem(
        'az',
        account_name=os.environ["STRG_ACCOUNT_NAME"] ,
        account_key=os.environ["STRG_ACCOUNT_KEY"] 
    )
    # filter to points we have response vars for
    pts = pts.join(response_vars, how='inner' if args.only_labeled_samples else 'left')

    # add column for whether the non-forest data is validated
    labeled_non_forest = pd.read_csv(args.non_forest_csv).set_index('INDEX')
    labeled_non_forest = labeled_non_forest.drop(labeled_non_forest[labeled_non_forest.FORESTED == 1].index).FORESTED
    pts = pts.join(labeled_non_forest)

    hls_to_tiles = dict()
    for h in fs.ls('fia/hls/chips/'):
        [idx, tile] = h.split('/')[-1].split('-')
        tile = tile[:5]
        if int(idx) not in hls_to_tiles:
            hls_to_tiles[int(idx)] = tile
    hls_df = pd.DataFrame(hls_to_tiles.items()).rename({0: 'INDEX', 1: 'tile'}, axis=1).set_index('INDEX')
    pts = pts.join(hls_df)

    if args.all_vars == 1:
        # add NAIP data
        naip_df = pd.DataFrame([
            int(n.split('/')[-1].split('.')[0])
            for n in fs.ls('fia/naip/')
        ]).rename({0: 'INDEX'}, axis=1).set_index('INDEX')
        naip_df['naip'] = True
        pts = pts.join(naip_df)


        # add DEM data
        dem_idxs = xr.open_zarr(
            fsspec.get_mapper(
                "az://fia/nasadem/nasadem-samples.zarr",
                account_name="usfs",
                account_key=args.account_key
            )
        ).coords['idx'].values
        dem_df = pd.DataFrame(dem_idxs).set_index(0)
        dem_df['dem'] = True
        pts = pts.join(dem_df)
    
    if not os.path.exists('gold_data_output'):
        os.makedirs('gold_data_output')
    pts.to_csv(f"gold_data_output/{args.output_csv}")

    # upload output csv to aml workspace
    fiads = Dataset.File.upload_directory(src_dir='./gold_data_output',
           target=DataPath(blob_datastore,  path_on_datastore='gold_data'))
    fiads.register(workspace=ws,
                   name='gold_data',
                   description='Chimera Training Data')
