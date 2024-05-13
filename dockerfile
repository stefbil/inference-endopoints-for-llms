# CERTH-ITI-VCL3D
# Authors: {petros.drakoulis, stefbil, atdovas}@iti.gr

# INHERIT FROM BASE IMAGE
FROM nvcr.io/nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

# INSTALL PYTHON AND PIP
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && sudo apt upgrade -y; exit 0
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.10-dev
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.10 1
RUN apt install curl -y
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# INSTALL THE REST OF THE PACKAGES
RUN apt install -y iputils-ping
RUN apt install -y nano
RUN apt install -y unzip
RUN apt install -y net-tools

# INSTALL PYTHON PACKAGES
ADD code /code
RUN pip install --upgrade pip
RUN pip install -r /code/requirements.txt


# COPY MODELS INTO ROOT OF CONTAINER
RUN mkdir -p -v root/.cache/
COPY huggingface* root/.cache/huggingface
COPY torch* root/.cache/torch

# MAKE CONTAINER FILESYSTEM
RUN mkdir action
ADD exec /action

# PORT-FORWARDING
EXPOSE 5035

# SET EXECUTION STARTING POINT
RUN chmod +rx /action/exec
ENTRYPOINT ["bash", "/action/exec"]
