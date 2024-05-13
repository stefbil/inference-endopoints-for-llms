from typing import Union
from fastapi import FastAPI, Response, UploadFile, Request, Form
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from transformers import LxmertForQuestionAnswering, LxmertTokenizer, AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel
from transformers import pipeline
from transformers import BlipProcessor, BlipForQuestionAnswering
import json
from utils import Config
from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess
# import subprocess
import numpy as np
import tempfile
import os
import shutil
import uvicorn
from datetime import datetime
import csv
import demo
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import pnpv2
import re
import image_caption
import vilt_vqa
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load models and model components for LXMERT
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)
lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
lxmert_gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased")
lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")

# Image Caption Hugging face
encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)

# load models and model components for PnP
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("................................model start loading................................")
model, vis_processors, txt_processors = load_model_and_preprocess(name="pnp_vqa", model_type="base", is_eval=True, device=device)
print("................................model loaded................................")

processor_vqa = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model_vqa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")


imate_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")




class Questions(BaseModel):
    file: UploadFile
    questions: str


@app.get("/")
async def read_root() -> dict:
    return {"message": "Welcome to Certh's API testing! Append to /docs to start using the api!"}

@app.post("/vqa_pnp/")
async def create_upload_file(file:UploadFile, questions: str):

    # create a new folder with the current date and time
    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")
    folder_name = now.strftime("%Y-%m-%d")
    folder_path = os.path.join("PnP VQA", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Split the filename and extension
    filename, ext = os.path.splitext(file.filename)
    new_filename = f"{filename}-{date_time}{ext}"
    
    # save the uploaded file to the new folder
    file_path = os.path.join(folder_path, new_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    #split the questions and get the answers
    # print(questions)
    #  questions = questions.lower()
    # questions = str(questions)
    # questions = questions.split(";")
    # print(questions)
    answers = pnpv2.pnpvqa(file_path, questions)

    # write the questions and answers to a csv file
    csv_path = os.path.join(folder_path, f"{filename}-{date_time}.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['Question', 'Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Question': questions, 'Answer': answers})
        # for i, q in enumerate(questions):
        #     writer.writerow({'Question': questions, 'Answer': answers})
    

    print(answers)
    # print(file_path)
    # os.remove(file_path)  
    return ({"Answer by PnP": answers })


@app.post("/cap_pnp/")
async def create_upload_file(file:UploadFile):

    # create a new folder with the current date and time
    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")
    folder_name = now.strftime("%Y-%m-%d")
    folder_path = os.path.join("PnP CAP", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Split the filename and extension
    filename, ext = os.path.splitext(file.filename)
    new_filename = f"{filename}-{date_time}{ext}"
    
    # save the uploaded file to the new folder
    file_path = os.path.join(folder_path, new_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    #split the questions and get the answers
    #  questions = questions.lower()
    # questions = str(questions)
    # questions = questions.split(";")
    # print(questions)
    caption = pnpv2.pnpcap(file_path)

    # write the questions and answers to a csv file
    csv_path = os.path.join(folder_path, f"{filename}-{date_time}.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['Image', 'Caption']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Image': new_filename, 'Caption': caption})
        # for i, q in enumerate(questions):
        #     writer.writerow({'Question': questions, 'Answer': answers})
    

    print(caption)
    # print(file_path)
    # os.remove(file_path)  
    return ({"Caption by PnP": caption })


@app.post("/vqa_lxmert/")
async def create_upload_file(file:UploadFile, questions: str):

    # create a new folder with the current date and time
    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")
    folder_name = now.strftime("%Y-%m-%d")
    folder_path = os.path.join("Lxmert VQA", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Split the filename and extension
    filename, ext = os.path.splitext(file.filename)
    new_filename = f"{filename}-{date_time}{ext}"
    
    # save the uploaded file to the new folder
    file_path = os.path.join(folder_path, new_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    #split the questions and get the answers
    questions = questions.split(";")
    pred_answers = demo.vqalxmert(file_path, questions)

    # write the questions and answers to a csv file
    csv_path = os.path.join(folder_path, f"{filename}-{date_time}.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['Question', 'Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, q in enumerate(questions):
            writer.writerow({'Question': q, 'Answer': pred_answers[i]})
    
    print(pred_answers)
    # os.remove(file_path)  
    return ({"Prediction by LXMERT": pred_answers})

@app.post("/cap_lxmert/")
async def create_upload_file(file:UploadFile):

    # create a new folder with the current date and time
    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")
    folder_name = now.strftime("%Y-%m-%d")
    folder_path = os.path.join("Lxmert CAP", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Split the filename and extension
    filename, ext = os.path.splitext(file.filename)
    new_filename = f"{filename}-{date_time}{ext}"
    
    # save the uploaded file to the new folder
    file_path = os.path.join(folder_path, new_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    #split the questions and get the answers
    pred_answers = demo.caplxmert(file_path)

    # write the questions and answers to a csv file
    csv_path = os.path.join(folder_path, f"{filename}-{date_time}.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['Image', 'Caption']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Image': new_filename, 'Caption': pred_answers})
    
    # os.remove(file_path)  
    return ({"Prediction by LXMERT": pred_answers})


@app.post("/cap_gpt2/")
async def create_upload_file(file:UploadFile):

    # create a new folder with the current date and time
    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")
    folder_name = now.strftime("%Y-%m-%d")
    folder_path = os.path.join("Caption", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Split the filename and extension
    filename, ext = os.path.splitext(file.filename)
    new_filename = f"{filename}-{date_time}{ext}"
    
    # save the uploaded file to the new folder
    file_path = os.path.join(folder_path, new_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    caption = image_caption.img_cap(file_path)

    # write the questions and answers to a csv file
    csv_path = os.path.join(folder_path, f"{filename}-{date_time}.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['Image', 'Caption']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Image': new_filename, 'Caption': caption})
    
    print(caption)
    caption = caption[0]['generated_text']
    # print(file_path)
    # os.remove(file_path)  
    return ({"Image caption": caption })


@app.post("/vqa_blip")
async def create_upload_file(file:UploadFile, questions: str):
    
    # create a new folder with the current date and time
    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")
    folder_name = now.strftime("%Y-%m-%d")
    folder_path = os.path.join("Blip VQA", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Split the filename and extension
    filename, ext = os.path.splitext(file.filename)
    new_filename = f"{filename}-{date_time}{ext}"
    
    # save the uploaded file to the new folder
    file_path = os.path.join(folder_path, new_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    #split the questions and get the answers
    questions = questions.split(";")
    pred_answers = vilt_vqa.vqa(file_path, questions)

    # write the questions and answers to a csv file
    csv_path = os.path.join(folder_path, f"{filename}-{date_time}.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['Question', 'Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, q in enumerate(questions):
            writer.writerow({'Question': q, 'Answer': pred_answers})
    
    print(pred_answers)
    # os.remove(file_path)  
    return ({"Answer by blip": pred_answers})


# if __name__ == "__main__":
#     uvicorn.run(app, host="195.251.117.32", port=5035)

#     uvicorn main:app --host 195.251.117.32 --port 5035
