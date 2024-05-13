# from transformers import pipeline
import time
import main


def img_cap(path):

    time1 = time.time()

    # image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

    prediction = main.imate_to_text(path)

    time2 = time.time()
    print("Total time in seconds:", time2-time1)


    return prediction