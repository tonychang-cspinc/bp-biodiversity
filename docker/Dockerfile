FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04:20201112.v1

RUN pip install \
    azureml-core==1.18.0.post1 \
    azureml-defaults==1.18.0 \
    azureml-telemetry==1.18.0 \
    azureml-train-restclients-hyperdrive==1.18.0 \
    azureml-train-core==1.18.0 \
    cmake==3.18.2 \
    torch==1.6.0 \
    torchvision==0.7.0 \
    mkl==2018.0.3 \
    tensorboard==1.14.0 \
    future==0.17.1 \
    adlfs==0.6.3 \
    fsspec==0.8.7 \
    pandas==1.1.5 \
    rasterio==1.2.0 \
    xarray==0.16.2 \
    zarr==2.6.1 \
    fiona==1.8.18 \
    affine==2.3.0 \
    pyyaml==5.4.1 \
    richdem==0.3.4 \
    netcdf4==1.5.6 \
    rtree==0.9.7 \
    shapely==1.7.1 \
    rioxarray==0.3.1 \
    rio-tiler==2.1.0 \
    rio-cogeo==2.2.1 \
    mercantile==1.2.1 \
    geopandas==0.10.2

RUN pip install horovod==0.21.3
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

CMD /bin/bash
