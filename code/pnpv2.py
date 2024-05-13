import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import time
import main

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# print("................................model start loading................................")
# model, vis_processors, txt_processors = load_model_and_preprocess(name="pnp_vqa", model_type="base", is_eval=True, device=device)
# print("................................model loaded................................")


def pnpvqa(path, question):

    time1 = time.time()

    # image_path = path
    raw_image = Image.open(path).convert('RGB')
    # question = questions
    # print(question)

    #image and text preprocessing
    image = main.vis_processors["eval"](raw_image).unsqueeze(0).to(main.device)
    question = main.txt_processors["eval"](question)
    samples = {"image": image, "text_input": [question]}


    # Relevancy score of image patches with respect to the question using GradCAM
    samples = main.model.forward_itm(samples=samples)


    # VQA
    pred_answers, caption, gradcam = main.model.predict_answers(samples, num_captions=50, num_patches=20)
    print('Question: {} \nPredicted answer: {}'.format(question, pred_answers[0]))
    # print('captions: ',caption[0][0])


    time2 = time.time()
    print("Total time in seconds:", time2-time1)
    # pred_answers = model.forward_qa(samples, num_captions=50)
    # print('Question: {} \nPredicted answer: {}'.format(question, pred_answers[0]))

    return pred_answers[0]



def pnpcap(path):
    
    
    time1 = time.time()

    question = "What is in the image?"

    # image_path = path
    raw_image = Image.open(path).convert('RGB')
    # question = questions
    # print(question)

    #image and text preprocessing
    image = main.vis_processors["eval"](raw_image).unsqueeze(0).to(main.device)
    question = main.txt_processors["eval"](question)
    samples = {"image": image, "text_input": [question]}


    # Relevancy score of image patches with respect to the question using GradCAM
    samples = main.model.forward_itm(samples=samples)


    # VQA
    samples = main.model.forward_cap(samples=samples, num_captions=50, num_patches=20)
    print('Examples of question-guided captions: ')
    samples['captions'][0][:1]


    time2 = time.time()
    print("Total time in seconds:", time2-time1)

    return samples['captions'][0][:1]