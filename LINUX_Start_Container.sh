#!/bin/bash
sudo docker run -it -p 5035:5035 --network host --gpus all --name pdrak_voxreality_all pdrak/voxreality:aio
