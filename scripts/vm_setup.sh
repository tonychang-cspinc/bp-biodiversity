#!/bin/bash
apt-get update
apt-get install -y git docker.io vim curl wget

##install nvidia drivers
## get dependencies first
apt-get install -y build-essential cmake unzip pkg-config \
	libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev\
	libjpeg-dev libpng-dev libtiff-dev \
	libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
	libxvidcore-dev libx264-dev libgtk-3-dev \
	libopenblas-dev libatlas-base-dev liblapack-dev gfortran \
	libhdf5-serial-dev python3-dev python3-tk python-imaging-tk \
	gcc-6 g++-6 \
	p7zip-full 
##install nvidia-driver
add-apt-repository -y ppa:graphics-drivers/ppa
apt-get update
apt install -y nvidia-driver-460
## install cuda
#wget -c https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb &&\
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 

wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-ubuntu1804-11-2-local_11.2.0-460.27.04-1_amd64.deb

dpkg -i cuda-repo-ubuntu1804-11-2-local_11.2.0-460.27.04-1_amd64.deb

apt-key add /var/cuda-repo-ubuntu1804-11-2-local/7fa2af80.pub

rm cuda-repo-ubuntu1804-11-2-local_11.2.0-460.27.04-1_amd64.deb

apt-get update
apt-get install -y cuda
echo 'export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
ldconfig
## install cudnn 7.6
## need to download cudnn from NVIDIA with registration code
wget -c https://storage.googleapis.com/csp-resources/cudnn-11.2-linux-x64-v8.1.0.77.tgz
tar -xvzf cudnn-11.2-linux-x64-v8.1.0.77.tgz  
rm cudnn-11.2-linux-x64-v8.1.0.77.tgz 
cp -P cuda/include/cudnn.h /usr/local/cuda-11.2/include
cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.2/lib64
chmod a+r /usr/local/cuda-11.2/lib64/libcudnn*
rm -r cuda
##install nvidia-docker2.0
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-docker2
pkill -SIGHUP dockerd
#edited 11/1/21 below, add azure cli
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash