@echo off
setlocal enabledelayedexpansion

docker stop llms
set FOLDER=%DATE:/=_%_%TIME::=_%
mkdir "Dumps\%FOLDER%"
docker cp llms:/code/Caption "Dumps\%FOLDER%"
docker cp llms:/code/PnP VQA "Dumps\%FOLDER%"
docker cp llms:/code/PnP CAP "Dumps\%FOLDER%"
docker cp llms:/code/Lxmert VQA "Dumps\%FOLDER%"
docker cp llms:/code/Lxmert CAP "Dumps\%FOLDER%"
docker cp llms:/code/Blip VQA "Dumps\%FOLDER%"
docker rm llms
