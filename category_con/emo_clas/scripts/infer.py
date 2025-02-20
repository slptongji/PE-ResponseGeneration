
import json, argparse, sys
import pandas as pd
import sys
import os

import torch

from main import *
import yaml

type_data = 'valid'

config_path = '../finetuned_llm/bioserc_bert_based/version_13/hparams.yaml'
model_path = "../finetuned_llm/bioserc_bert_based/roberta-large-meld-valid/f1=67.41.ckpt"

print('config_path', config_path)

with open(config_path, "r") as yamlfile:
    model_configs = argparse.Namespace(**yaml.load(yamlfile, Loader=yaml.FullLoader))

model_configs.data_folder = '../data/'
# model_configs.window_ct = 2
# model_configs.speaker_description = False
# model_configs.llm_context = False
# model_configs.data_name_pattern = "meld.{}.json"
dataset_name = model_configs.data_name_pattern.split(".")[0]




# meld
label2id = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
id2label = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']


# # emorynlp
# label2id = {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
# id2label = ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared']
# # iemocap
# label2id = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
# id2label = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']



from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import json
import random
import argparse
from sklearn.metrics import f1_score

import pytorch_lightning as pl
from pytorch_lightning import Trainer

bert_tokenizer  = AutoTokenizer.from_pretrained("../finetuned_llm/roberta")

# 验证集的数据加载器
'''
data_loader_valid = BatchPreprocessor(bert_tokenizer, model_configs=model_configs, data_type=type_data)
raw_data = BatchPreprocessor.load_raw_data(f"{model_configs.data_folder}/{model_configs.data_name_pattern.format(type_data)}")
valid_loader = DataLoader(raw_data,
                          batch_size=model_configs.batch_size, collate_fn=data_loader_valid, shuffle=False)
'''
# 测试集的数据加载器
data_loader_test=BatchPreprocessor(bert_tokenizer, model_configs=model_configs, data_type='test')
raw_data_test = BatchPreprocessor.load_raw_data(f"{model_configs.data_folder}/output_meld_format_valid.json")
    #(f"{model_configs.data_folder}/{model_configs.data_name_pattern.format('test')}")
test_loader = DataLoader(raw_data_test,
                          batch_size=model_configs.batch_size, collate_fn=data_loader_test, shuffle=False)


model_configs.spdesc_aggregate_method = 'static'
# model_configs.llm_context = False
# model_configs.speaker_description=False

#model = EmotionClassifier(model_configs)
# model.model_configs = model_configs


import json
import itertools
# 加载模型
model = EmotionClassifier.load_from_checkpoint(model_path, strict=False, model_configs=model_configs)
trainer = Trainer(max_epochs=1,  accelerator="gpu", devices=1,  )
# 创建训练器并进行测试
#print(trainer.test(model, valid_loader))
print(trainer.test(model, test_loader))

