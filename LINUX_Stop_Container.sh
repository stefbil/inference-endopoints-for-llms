#!/bin/bash
sudo docker llms
FOLDER=$(date +%Y_%m_%d_%r)
mkdir -p Dumps
sudo docker cp "llms:/code/Blip VQA" "./Dumps/$FOLDER"
sudo docker cp "llms:/code/Lxmert VQA" "./Dumps/$FOLDER"
sudo docker cp "llms:/code/Caption" "./Dumps/$FOLDER"
sudo docker cp "llms:/code/PnP CAP" "./Dumps/$FOLDER"
sudo docker cp "llms:/code/Lxmert CAP" "./Dumps/$FOLDER"
sudo docker cp "llms:/code/PnP VQA" "./Dumps/$FOLDER"
sudo docker rm llms
