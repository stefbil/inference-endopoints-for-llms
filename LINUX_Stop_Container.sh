#!/bin/bash
sudo docker pdrak_voxreality_all
FOLDER=$(date +%Y_%m_%d_%r)
mkdir -p Dumps
sudo docker cp "pdrak_voxreality_all:/code/Blip VQA" "./Dumps/$FOLDER"
sudo docker cp "pdrak_voxreality_all:/code/Lxmert VQA" "./Dumps/$FOLDER"
sudo docker cp "pdrak_voxreality_all:/code/Caption" "./Dumps/$FOLDER"
sudo docker cp "pdrak_voxreality_all:/code/PnP CAP" "./Dumps/$FOLDER"
sudo docker cp "pdrak_voxreality_all:/code/Lxmert CAP" "./Dumps/$FOLDER"
sudo docker cp "pdrak_voxreality_all:/code/PnP VQA" "./Dumps/$FOLDER"
sudo docker rm pdrak_voxreality_all
