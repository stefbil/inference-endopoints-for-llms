import requests
from PIL import Image
# from transformers import BlipProcessor, BlipForQuestionAnswering
import time
import main

# processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
# model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")


def vqa(path, question):
    time1 = time.time()

    # img_url = 'https://cdn.discordapp.com/attachments/1051862043021541449/1082959999808045076/image.png' 
    raw_image = Image.open(path).convert('RGB')

    # question = "What are the ingredients on the pizza?"
    inputs =main.processor_vqa(raw_image, question, return_tensors="pt").to("cuda")
    out = main.model_vqa.generate(**inputs)
    # print(main.processor_vqa.decode(out[0], skip_special_tokens=True))

    time2 = time.time()
    print(time2-time1)

    return main.processor_vqa.decode(out[0], skip_special_tokens=True)