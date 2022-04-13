#!/bin/bash

# Uses the same docker image as traing_docker.sh, but runs with GPU capabilities and hardware resource allocation explicitly defined.

[[ -z "${ACR_IMAGENAME}" ]] && echo 'Please edit and run set_session_vars.sh' && exit
checkim=$( { sudo docker image inspect $ACR_REGISTRY.azurecr.io/$ACR_IMAGENAME:$ACR_IMAGETAG; } 2>&1)
_DIMG_EXISTS=${checkim:3:8}

if [ ${_DIMG_EXISTS:0:5} = "Error" ]; then
    echo 'Image not found locally. Building and uploading...'
    sudo az login
    sudo az acr login --name $ACR_REGISTRY
    sudo az acr build --image $ACR_IMAGENAME:$ACR_IMAGETAG \
                --registry $ACR_REGISTRY \
                --resource-group $ACR_RESOURCE_GROUP \
                --file docker/Dockerfile \
                docker/
else
    echo "Image found locally."
fi

sudo nvidia-docker run -it --rm --privileged \
    -p $DOCKER_PORT:$DOCKER_PORT \
    -v $('pwd'):/content \
    -m 100g \
    --shm-size=20g \
    $ACR_REGISTRY.azurecr.io/$ACR_IMAGENAME:$ACR_IMAGETAG \
    bash -c "source ./content/scripts/set_session_vars.sh; cd content; bash"