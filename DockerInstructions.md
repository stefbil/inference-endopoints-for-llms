# Docker Instructions for the inference-endpoints-for-llms project

Below you'll find the necessary instructions in order to download and run the docker images.


## 1. Requirements
---
1. CUDA compatible GPU with at least 12GB of VRAM
2. For LINUX (tested on Ubuntu 20.04)
   1. The system was tested on Nvidia proprietary driver 515 and 525
   2. Make sure Docker is installed on your system. For instructions you can refer to the [official docker guide](https://docs.docker.com/desktop/install/ubuntu/)
   3. Make sure you have the NVIDIA Container Toolking installed. More info and instructions can be found in the [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
3. For Windows
   1. TO BE ADDED

One you have docker up and running you can move to downloading the image.

## 2. Downloading the image
---
1. Start by pulling the desired docker image 
   1. `sudo docker pull stefbil/inference-endopoints-for-llms:v1`
2. Makes sure that the image appears on your system by running `sudo docker images` 

## 3. Running the docker containter
---
1. Run the docker container by executing the following command:
   1. `sudo docker run -it -p 5035:5035 --network host --gpus all --name llms stefbil/inference-endopoints-for-llms:v1`
   Note that the `-p` argument exposes the apropriate port to listening and receiving events. The `--name` is the container name which can be changed at will.
2. The container should start downloading the necessary models in order to run properly (if they aren't present).
3. Once finished (many minutes, depening on internet speed), you should see a message saying: `Uvicorn running on http://[HOST PC Public IP]:PORT`

## 4. Using the api
---
### 4.1 Using REST calls (Postman)
The API exposes 6 endpoints serving 4 models for accomplishing 2 vision-language tasks: 

1. Visual Question Answering (/vqa_x): A multimodal task wherein, given an image and a natural language question related to the image, the objective is to produce a natural language answer as output.
2. Image-Captioning (/cap_x): The task of translating an input image into a textual description.

The REST endpoints are:

1. `/vqa_pnp` : Visual Question Answreing task using the PNP model.
2. `/cap_pnp` : Image-captioning task using the PNP model.
3. `/vqa_lxmert` : Visual Question Answreing task using the lxmert model.
4. `/cap_lxmert` : Image-captioning task using the LXMERT model.
5. `/cap_gpt2` : Image-captioning task using the ViT_GPT2 model.
6. `/vqa_blip` : Visual Question Answreing task using the BLIP model.

You can see an example of calling a `/vqa_x` endpoint via REST call using Postman below:

   1. Create a new tab and set the request type to POST
   2. Type in the `HOST-IP address` and `port`, followed by the prefered vqa task endpoint. 

   ![vqa postman](https://cdn.discordapp.com/attachments/1050414414488162345/1084783347533893752/image.png)
   3. Open the `Params` tab below, and check the first row.

   4. For the `KEY` column type in `questions`. For the `VALUE` column type in `your question`.

   ![](https://cdn.discordapp.com/attachments/1050414414488162345/1084782059161141268/image.png)
   5. Move to the `Body` tab, select the `form-data` round-box and check the first row.

   6. Set the `KEY` type to `file` and type `file` in the textbox. Upload your desired file to the `VALUE` column and set the `CONTENT TYPE` column to `multipart/form-data`.

   ![](https://cdn.discordapp.com/attachments/1050414414488162345/1084782156364132382/image.png)

   7. Hit the `SEND` button and get your answer.

You can see an example of calling a `/cap_x` endpoint via REST call using Postman below:

   1. Create a new tab and set the request type to POST
   2. Type in the `HOST-IP address` and `port`, followed by the prefered cap task endpoint.

   ![cap postman](https://cdn.discordapp.com/attachments/1050414414488162345/1084781949412987021/image.png)

   3. Move to the `Body` tab, select the `form-data` round-box and check the first row.

   4. Set the `KEY` type to `file` and type `file` in the textbox. Upload your desired file to the `VALUE` column and set the `CONTENT TYPE` column to `multipart/form-data`.

   ![](https://cdn.discordapp.com/attachments/1050414414488162345/1084782156364132382/image.png)

   5. Hit the `SEND` button and caption.

### 4.2 Using the web-interface
For convenience and demonstration purposes, the whole system's functionality is available via a web-interface. To use it:

1. Navigate to the following address: `http://[HOST PC Public IP]:5035/docs`
2. Select any of the POST requests by clicking on them

![](https://cdn.discordapp.com/attachments/1050414414488162345/1084789856858800199/image.png)

3. Click on the button saying `Try it out` on the top right corner of each opened POST block

![](https://cdn.discordapp.com/attachments/1050414414488162345/1084790775046488074/image.png)
4. Depending on your selected task, input your question and/or your image

![](https://cdn.discordapp.com/attachments/1050414414488162345/1084790920689487973/image.png)

5. Click the blue button saying `Execute` below

![](https://cdn.discordapp.com/attachments/1050414414488162345/1084791005284413471/image.png)

6. Scroll down and get your answer on the Responses part of the opened POST block 

![](https://cdn.discordapp.com/attachments/1050414414488162345/1084791119260438588/image.png)

## 5. Useful info
---
1. It is recommended to use a GPU of at least 12GBs of VRAM, but that is the peak usage we measured by using every single model included
2. The image that uses all 6 endpoints at once (pdrak/voxreality:1) has the following requirements:
   1.  The api uses around 7 - 8 GBs of VRAM when it first loads and stays idle
   2.  Even if you don't own a GPU with 12GBs of VRAM, you can still use the api by using *some* of the models provided below
       1.  By using `cap_gpt2` the dedicated GPU memory is expected to rise by at least 1.5GBs and may peak at +1.7GBs for a short period of time
       2.  By using `vqa_blip` the dedicated GPU memory is expected to rise by at least 1.5GBs
       3.  By using `vqa_pnp` and `cap_pnp` the dedicated GPU memory is expected to rise by at least 4GBs and may peak at +3.7GBs for a short period of time
       4.  By using `vqa_lxmert` and `cap_lxmert` the dedicated GPU memory is not affected (in this case)
3. The image that only serves the lxmert model (pdrak/voxreality:lxmert) will use around 2-2.5GBS of VRAM.
4. The image that only serves the GPT2 model (pdrak/voxreality:gpt2) will use around 2-2.5GBS of VRAM.
5.  The api serves the models developed in 4 SoTA projects:
    1.  Here's the [code](https://github.com/salesforce/LAVIS/tree/main/projects/pnp-vqa) for the [PNP-VQA paper](https://arxiv.org/abs/2210.08773)
    2.  Here's the [code](https://github.com/huggingface/transformers/tree/main/examples/research_projects/lxmert) for the [LXMERT paper](https://arxiv.org/abs/1908.07490)
    3.  Here's the [ViT_GPT2 model](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) provided by [NLP Connect](https://github.com/nlpconnect)
    4.  Here's the [BLIP model](https://huggingface.co/Salesforce/blip-vqa-base) from the [paper](https://arxiv.org/abs/2201.12086)
