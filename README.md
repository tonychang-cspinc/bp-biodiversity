
# CSP x AI For Earth - Deep Forest Disturbance Analysis

## Introduction

Forest structure is a vital metric in ecological studies for delineating habitat location, condition, vulnerability, and connectivity for numerous species and processes of concern. There are currently few, if any, near-real time metrics of forest structure available to the conservation and management communities. We propose to utilize the Azure cloud platform and advanced deep learning methods to produce a forest structure monitoring application for the United States that readily updates annually, after new high-resolution imagery acquisitions are made available. This application would differ from other efforts to estimate and monitor forest structure--principally--through our unique relationship with the USFS to: 1) utilize the Forest Inventory and Analysis (FIA) dataset to train a deep learning model across various ecotypes and then 2) predict wall-to-wall across space and time within the US to identify changes on the forested landscape. Outputs would help in the identification of areas of highest importance for conservation and allow deeper insights into how human or natural disturbances over time are driving, for example, multiscale patterns of habitat fragmentation, species or carbon loss, water resource availability, or fire risk. 

## Modeling forest structure
This repository provides code for:
1. Training neural networks for modeling forest structure metrics and predicting forest type
2. Predicting forest structure metrics and forest type using trained model(s) from (1)
3. Preparing predictions from (2) for display on maps

We run all of our training, prediction, and other processing code using various Azure technologies so make sure you have an Azure account and credits, or are prepared to translate these steps to a different platform.
* All steps: Azure Container Registry (ACR), Blob storage
* Data pre-processing (https://github.com/csp-inc/data-ingestion)
  * The steps below assume you have already followed the steps in this repo to collect pre-processed data for both training and prediction.
* Training: AzureML (GPU Cluster)
* Prediction: Azure VMs (NC24v3)
* Display: Azure App Services, Azure VMs (pre-processing)


# Guide

### Environment Setup

For all steps through the creation of a tiles .csv for predictions (`05-build-prediction-csv.py`), we recommend using a lightweight Ubuntu 18.04 [Azure VM](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-portal), such as a D4s_v4 or D4s_v3 with at least 64gb disk storage.  

**Be sure to create this VM in the same region as your preprocessed data storage. Also be sure to create the VM in a region and availability zone with access to GPU (NC Series) instances.**  

Begin by [cloning or forking this repo](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) from a terminal on the VM, and then enter the root directory.

### Docker

The two following steps will take several minutes to run.

Run the VM setup script to install the needed software for GPU computing (used later for prediction), Docker, and the Azure CLI.
```
sudo bash scripts/vm_setup.sh
```
If you don't yet have a container registry on Azure you can follow the [ACR quickstart](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-azure-cli) to create one. 

To build our docker image, upload it to the ACR, and run it interactively, **edit the file /scripts/set_session_vars.sh** with your desired values and run it from the root directory.

Images showing where to find several variables can be found in the images directory.
```
source ./scripts/set_session_vars.sh
bash scripts/training_docker.sh
```
You will be prompted to log in to your azure account. Follow the instructions in the terminal.


The rest of the steps below will be run within an interactive Docker environment.

### Creating AzureML Workspace

Simply run the `01-create-workspace.py` python script. Example:
```
python 01-create-workspace.py \
    --name="my-cool-workspace" \
    --region="eastus"
```

Make sure to create the workspace in the same region as your preprocessed data. See the link [here](https://docs.microsoft.com/en-us/azure/machine-learning/) for more information on the capabilities and setup of AzureML workspaces.

**You may be prompted to log in once more at this step.**

This will produce a config file in a new .azureml directory that will be used in subsequent steps. 

### Create AzureML cluster(s)

In order to train in your new ML workspace you need a cluster. This script will generate a new cluster of up to max-nodes of vm-size in your workspace.

Simply run the `02-create-compute.py` script. Example:
```
python 02-create-compute.py \
    --name="gpu-cluster" \
    --vm-size="STANDARD_NC6" \
    --max-nodes=6
```

### Create gold data for training

The training script expects a CSV file containing rows describing each training sample. Each row needs to have the following columns:
* INDEX: the sample index
* INVYR: year sample was taken
* STATECD: state code
* LAT: latitude
* LON: longitude
* TPA: Trees per acre
* BAA: basal area per acre
* CARB_ACRE: carbon per acre
* nStems: num stems
* BIO_ACRE: biomass per acre
* PLOT_STATUS: Forested / Non-forested (non-forested may have trees)
* CLASS: 0 = None, 1 = Conifer, 2 = Deciduous, 3 = Mixed, 4 = Dead
* CANOPY_CVR: Percent canopy cover
* BASAL_AREA: basal area per acre
* ECO_US_ID: Not used
* ECOCODE: bailey's ecocode
* DOMAIN: bailey's ecoregion domain
* DIVISION: bailey's ecoregion division
* PROVINCE: bailey's ecoregion province
* SECTION: bailey's ecoregion section
* FORESTED: whether a non-forested plot actually has trees (1 or 0)
* tile: HLS/MGRP tile for sample

The following two columns can be omitted by passing all-vars 0, speeding up the process
* naip: True/False: do we have naip data for sample
* dem: True/False - do we have NASADEM data for the sample

This csv file can be generated by running `03-create-gold-data-csv.py`. The script ingests three master CSVs:

[fia_no_pltctn.csv](https://usfs.blob.core.windows.net/fia/fia_no_pltcn.csv), [fia_ytrain.csv](https://usfs.blob.core.windows.net/fia/fia_ytrain.csv), and [nonlabeled-non-forest.csv](https://usfs.blob.core.windows.net/fia/labeled-non-forest.csv) are maintained by CSP and available for download.

They can be downloaded to the local directory by typing:
```
bash scripts/download_csvs.sh
```
03-create-gold-data.py expects the blob virtual directory structure of preprocessed training samples to be as follows:

```
    $STRG_ACCOUNT_NAME      # defined in set_session_vars.sh or given as argument
    ├── $BLOB_CONTAINER     # defined in set_session_vars.sh or given as argument
    │   ├── naip            
    │   └── nasadem    
    │       └── nasadem-samples.zarr
    │   ├── hls             
    │   │     └── chips
    │   ├── **gold_data     # output_csv will be uploaded to new virtual directory here
    │   └── ...             
    └── ...
```

If the blob virtual directory structure is different than above (the default from the data-ingestion repository), modifications will be needed.

The output of this script will be a new datastore named 'fia', referencing the given $BLOB_CONTAINER, and a new dataset in that datastore named 'gold_data', both registered to the Machine Learning Workspace.

The registered dataset can be [updated and versioned](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-version-track-datasets#versioning-best-practice) as more FIA data is released.

Example:
```
python 03-create-gold-data-csv.py \
    --plot-csv=fia_no_pltcn.csv \
    --response-csv=fia_ytrain.csv \
    --output-csv=modeling_data_2.csv \
    --account-key=XXXX \
    --non-forest-csv=labeled-non-forest.csv
```

### Train!

Now we've got everything set up and we can train using AzureML.

Example:
```
python 04-train-net.py \
    --cluster="gpu-cluster"
    --sources-path="src/config/sources/hls.yml" \
    --params-path="src/config/hls/all.yml" \
```
For more information on these config files see their templates: [sources](/src/configs/sources/template.yml), [params](/src/configs/template.yml)

The sources and params yaml config files determine which model to train, on what data, and with what hyperparameters. 

More information on running this script can be found in the file itself and its sub-script: `src/train.py`.

Metrics and output files will appear in the AzureML run from your submission. Model results can be compared across different runs by editing [the view](/images/model_comparison.png).

After you are happy with model results, you can download the selected model as shown [here.](/images/model_output.png) Also take note of the [dataset max values](/images/dataset_maxes.png) printed in 70_driver_log_0.txt near the top of a successful run.

## Prediction

While training finishes, you can prepare your prediction parameters by building a csv of [HLS/MGRS tiles](https://hls.gsfc.nasa.gov/products-description/tiling-system/) and years that you want to predict on. If you developed an [eco-region](https://doi.org/10.2737/WO-GTR-76B) specific model in the steps above, we can specify tiles based on which province their bounding boxes fall in using the following script:

```
python 05-build-prediction-csv.py \
    --vector-path="eco_us.geojson" \
    --ecocode-prefixes "-24" "M24" "M26" "-26" \
    --years 2015 2016 2017 2018 2019 \
    -o predict_24_26_m26_m24.csv
```

The output from this script is a csv with two columns, "tile" and "year".

Here, eco_us.geojson is a [geojson](https://usfs.blob.core.windows.net/fia/eco_us_bgs.geojson) with ecoregions defined to the section level for the continental United States.

If ecocode-prefixes is not defined, the script returns all tile, year combinations that intersect the given geometries.

You may also make a tiles csv with the query_blob.py script.

Once you have a trained model you are satisfied with, you can use it to predict on a list of HLS tiles. 

First, we need to change the VM size. To do so, open the Azure portal, find the virtual machine you have been working in so far and stop it. **This will interrupt any connections and shut down the machine.**

After stopping, click the "size" button, and search for a desired GPU (NC-Series, preferably an NC24v3) instance.

Once the VM is resized, we can run the rest of the steps from within a GPU-enabled docker container. We also need to reset environment variables in the new shell.:

```
source ./scripts/set_session_vars.sh
bash scripts/prediction_docker.sh
```

** If you use a size other than NC24v3, the -m and --shm-size arguments in the docker run command will need to be rescaled appropriately, as well as the parameters at the top of 06-predict.py**


Example:
```
python 06-predict.py \
    -o output_dir \
    --model-path=path/to/model-file \
    --tiles-path=path/to/tiles-csv \
    --sources-path=src/configs/sources/hls.yml \
    --params-path=src/configs/hls/24-26-m24-m26-chip4.yml \
    -s 712.2573 685.2152 100.0000
```

* `-o` is the root dir for all predictions to be written into
* `--model-path` is the path to a model file, downloaded from AzureML
* `--tiles-path` is the path to a csv with two columns: `tile` which is the HLS tile ids to predict, and `year`. 
* `--sources-path` is the path to the same sources file that was used for training
	* **Note the "prediction parameters" section at the bottom of this file.**
* `--params-path` is the path to the same params file that was used for training
* `-s` specifies the scale multipliers for each prediction variable right now in order these are typically: basal area, bio acre, canopy cover. These values should be the max value from the training data and is logged during training so can be [found](/images/dataset_maxes.png) in your AzureML run for the model you are using.

Once prediction has finished the data is saved as COGs within sub-directories of your specified output directory. 

After QA of your predictions, you can upload this data to blob storage for access by the dynamic tile server discussed below. For example:
```
az storage azcopy blob upload -c {app/hls_cog} --account-name $STRG_ACCOUNT_NAME -s {prediction_output}  --recursive
```
This same command can be restructured for uploading output overview tiles and mosaicjsons discussed below.

## Display

### Dynamic Tile Server
In order to display predictions on a map we leverage https://github.com/developmentseed/titiler, a dynamic tile server.

We deploy our tiling server on Azure App Services using the [default docker image](https://github.com/developmentseed/titiler#docker) for titiler. Instructions for deploying a docker container on Azure App Services can be found in the [Azure docs](https://docs.microsoft.com/en-us/azure/app-service/tutorial-custom-container?pivots=container-linux).

For your preferred cloud deployment make sure the necessary environment variables for your titiler container are set:
* `API_CORS_ORIGINS` ([from here](https://github.com/developmentseed/titiler/blob/0ad40dbad6fce8ae1ae3582ff02a41baeda86fd6/titiler/application/titiler/application/settings.py#L10))
* `PORT` set to 80 (optional)
* `WORKERS_PER_CORE` - up to you

You may also want to include other performance tuning environment variables documented [here](https://devseed.com/titiler/advanced/performance_tuning/).

### Create MosaicJSONs

titiler requires [MosaicJSON](https://github.com/developmentseed/mosaicjson-spec) files for our use case. To create mosaicjsons for your prediction outputs run `07-create-mosaicjson.py`.

Example command:
```
python 07-create-mosaicjson.py \
    --base-path="https://usfs.blob.core.windows.net/app/hls_cog" \
    --tiles-path="tiles.csv" \
    -y 2016 2018 \
    -v class canopy_cvr basal_area bio_acre \
    --min-zoom=8 \
    --max-zoom=14
```
In this example:
* the base path is the path to the root directory where predictions are stored, predictions are expected to be stored at `{base_path}/{variable}/{year}`
* `tiles.csv` is a CSV containing a single column `tile` and contains a list of HLS tile ID `.tif`s that are stored in the sub-directories of the base path.
* -y specifies which years to produce mosaicjsons for
* -v specifies which variables to produce mosaicjsons for - the variables should be some subset of `class canopy_cvr basal_area bio_acre` at present.
* min-zoom - min zoom the mosaicjson should produce map tiles for
* max-zoom - max zoom the mosaicjson should produce map tiles for

Once this command is run mosaicjson files will be written locally (by default to `outputs/`). After the script is run the resulting files should be copied to blob storage in a publicly accessible blob container so they can be accessed by titiler.

### Create Overviews

The output predictions from this pipeline are 30m x 30m pixel resolution. For low zoom map tiles (<8) many HLS tiles have to be loaded by titiler in order to generate a map tile. This is CPU, IO and memory intensive so is inefficient and not performant. In order to mitigate this we create overview tiles that are web mercator (map) tile aligned tifs that are used at low zooms.

To generate low zooms run the `08-create-overviews.py` script. This script takes as input one of the mosaicjsons from the previous step and uses that data to generate mosaics at the specified zoom level for each valid map tile and save it as a COG at a particular resolution as an overview. It also generates a mosaicjson file for the overview tiles.

This script will read data from wherever your prediction tiles are stored so it is recommended that it be run in the same region as those tiles.

To generate overviews for all year and variable combinations that you have predicted in one run, modify and run scripts/all_overviews.sh.

Example command:
```
python 08-create-overviews.py \
-z 6 \
-m outputs/canopy_cvr-2018-z8-z14.json \
-o overviews/canopy_cvr/2018 \
-p "outputs/canopy_cvr-2018" \
-c "https://usfs.blob.core.windows.net/app/overview_cog/canopy_cvr/2018" \
--min-zoom=6 \
--max-zoom=7 \
-t 1024
```
In this example:
* -z specifies that we will generate zoom 6 tile tifs
* -m specifies the path to the mosaicjson to use to create the zoom 6 tile tifs
* -o specifies the output directory for the resulting tifs
* -p specifies the output prefix for the overview mosaicjson
* -c specifies the base path for overview tiles in the overview mosaicjson file.
* --min-zoom specifies the min zoom for the overview mosaicjson
* --max-zoom specifies the min zoom for the overview mosaicjson
* -t specifies the width/height in pixels for the output tile.

The results from this example can then be copied to blob storage (as in previous steps) using azcopy. They will then be referred to in urls discussed below.

### Visualizing predictions

Now that we have mosaicjsons, overviews, and titiler running we can efficiently generate map tiles on the fly.

See bio_acre_2019.html for a simple display example. Lines 97 and 118 can be modified with the appropriate URL format:

```
"https://{titiler-domain}/mosaicjson/tiles/{z}/{x}/{y}@2x.png?url={mosaicjson_blob_url}&rescale=0,{variable_max}&colormap_name=viridis
```

More info can be found in the titiler docs for the [mosaicjson endpoints](https://devseed.com/titiler/endpoints/mosaic/) and [cog endpoints](https://devseed.com/titiler/endpoints/cog/).

## Authors

* **Ben Shulman** - *Development & implementation*
* **Tony Chang** - *Concept & initial models*

## License

This project is not licensed

