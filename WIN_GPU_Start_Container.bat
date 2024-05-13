@echo off
docker run -it -p 5035:5035 --network bridge --gpus all --name voxreality_all voxreality/draft_vlmodels:all
