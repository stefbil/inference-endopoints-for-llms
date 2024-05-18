# %pip install-r requirements.txt
# from IPython.display import clear_output, Image, display

import PIL.Image
import io
import json
import torch
import numpy as np
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
# import wget
import pickle
import os
import sys
import main
from typing import Union


def vqalxmert(URL, questions):

    OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
    ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
    # GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
    VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"


    # # for visualizing output
    # def showarray(a, fmt="jpeg"):
    #     a = np.uint8(np.clip(a, 0, 255))
    #     f = io.BytesIO()
    #     PIL.Image.fromarray(a).save(f, fmt)
    #     display(Image(data=f.getvalue()))



    # load object, attribute, and answer labels

    objids = utils.get_data(OBJ_URL)
    attrids = utils.get_data(ATTR_URL)
    # gqa_answers = utils.get_data(GQA_URL)
    vqa_answers = utils.get_data(VQA_URL)

    # URL
    #     
    # frcnn_visualizer = SingleImageViz(URL, id2obj=objids, id2attr=attrids)
    # run frcnn
    images, sizes, scales_yx = main.image_preprocess(URL)
    output_dict = main.frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=main.frcnn_cfg.max_detections,
        return_tensors="pt",
    )

    # add boxes and labels to the image

    # frcnn_visualizer.draw_boxes(
    #     output_dict.get("boxes"),
    #     output_dict.pop("obj_ids"),
    #     output_dict.pop("obj_probs"),
    #     output_dict.pop("attr_ids"),
    #     output_dict.pop("attr_probs"),
    # )

    # showarray(frcnn_visualizer._get_buffer())



    # test_questions_for_url1 = [
    #     "Where is this scene?",
    #     "what is the man riding?",
    #     "What is the man wearing?",
    #     "What is the color of the horse?",
    # ]

    test_questions_for_url2 = questions
    # [
    #     "What is behind the laptop?",
    # ]

    # Very important that the boxes are normalized
    normalized_boxes = output_dict.get("normalized_boxes")
    features = output_dict.get("roi_features")

    pred_answers = []
    for test_question in questions:
        # run lxmert
        test_question = [test_question]
        # pred_answers = []

        inputs = main.lxmert_tokenizer(
            test_question,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # run lxmert(s)
        # output_gqa = main.lxmert_gqa(
        #     input_ids=inputs.input_ids,
        #     attention_mask=inputs.attention_mask,
        #     visual_feats=features,
        #     visual_pos=normalized_boxes,
        #     token_type_ids=inputs.token_type_ids,
        #     output_attentions=False,
        # )
        output_vqa = main.lxmert_vqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            output_attentions=False,
        )
        # get prediction
        pred_vqa = output_vqa["question_answering_score"].argmax(-1)
        # pred_gqa = output_gqa["question_answering_score"].argmax(-1)
        # print("Question:", test_question)
        # print("prediction from LXMERT GQA:", gqa_answers[pred_gqa])
        # print("prediction from LXMERT VQA:", vqa_answers[pred_vqa])
        pred_answers.append(vqa_answers[pred_vqa])
        # pred_answers = "".join(pred_answers)
        
    # print(pred_answers)
    return ([pred_answers])


def caplxmert(URL):

    OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
    ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
    # GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
    VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"


    # # for visualizing output
    # def showarray(a, fmt="jpeg"):
    #     a = np.uint8(np.clip(a, 0, 255))
    #     f = io.BytesIO()
    #     PIL.Image.fromarray(a).save(f, fmt)
    #     display(Image(data=f.getvalue()))



    # load object, attribute, and answer labels

    objids = utils.get_data(OBJ_URL)
    attrids = utils.get_data(ATTR_URL)
    # gqa_answers = utils.get_data(GQA_URL)
    vqa_answers = utils.get_data(VQA_URL)

    # URL
    #     
    # frcnn_visualizer = SingleImageViz(URL, id2obj=objids, id2attr=attrids)
    # run frcnn
    images, sizes, scales_yx = main.image_preprocess(URL)
    output_dict = main.frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=main.frcnn_cfg.max_detections,
        return_tensors="pt",
    )

    # add boxes and labels to the image

    # frcnn_visualizer.draw_boxes(
    #     output_dict.get("boxes"),
    #     output_dict.pop("obj_ids"),
    #     output_dict.pop("obj_probs"),
    #     output_dict.pop("attr_ids"),
    #     output_dict.pop("attr_probs"),
    # )

    # showarray(frcnn_visualizer._get_buffer())



    # test_questions_for_url1 = [
    #     "Where is this scene?",
    #     "what is the man riding?",
    #     "What is the man wearing?",
    #     "What is the color of the horse?",
    # ]

    questions = [
        "What is in the image?",
        "Describe the image?",
    ]

    # Very important that the boxes are normalized
    normalized_boxes = output_dict.get("normalized_boxes")
    features = output_dict.get("roi_features")

    pred_answers = []
    for test_question in questions:
        # run lxmert
        test_question = [test_question]
        # pred_answers = []

        inputs = main.lxmert_tokenizer(
            test_question,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # run lxmert(s)
        # output_gqa = main.lxmert_gqa(
        #     input_ids=inputs.input_ids,
        #     attention_mask=inputs.attention_mask,
        #     visual_feats=features,
        #     visual_pos=normalized_boxes,
        #     token_type_ids=inputs.token_type_ids,
        #     output_attentions=False,
        # )
        output_vqa = main.lxmert_vqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            output_attentions=False,
        )
        # get prediction
        pred_vqa = output_vqa["question_answering_score"].argmax(-1)
        # pred_gqa = output_gqa["question_answering_score"].argmax(-1)
        # print("Question:", test_question)
        # print("prediction from LXMERT GQA:", gqa_answers[pred_gqa])
        # print("prediction from LXMERT VQA:", vqa_answers[pred_vqa])
        pred_answers.append(vqa_answers[pred_vqa])
        # pred_answers = "".join(pred_answers)
        
    # print(pred_answers)
    return ([pred_answers])