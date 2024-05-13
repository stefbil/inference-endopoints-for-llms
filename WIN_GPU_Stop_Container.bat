@echo off
setlocal enabledelayedexpansion

docker stop voxreality_all
set FOLDER=%DATE:/=_%_%TIME::=_%
mkdir "Dumps\%FOLDER%"
docker cp voxreality_all:/code/Caption "Dumps\%FOLDER%"
docker cp voxreality_all:/code/PnP VQA "Dumps\%FOLDER%"
docker cp voxreality_all:/code/PnP CAP "Dumps\%FOLDER%"
docker cp voxreality_all:/code/Lxmert VQA "Dumps\%FOLDER%"
docker cp voxreality_all:/code/Lxmert CAP "Dumps\%FOLDER%"
docker cp voxreality_all:/code/Blip VQA "Dumps\%FOLDER%"
docker rm voxreality_all
