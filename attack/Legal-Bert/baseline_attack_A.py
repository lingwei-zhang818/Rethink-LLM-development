import pandas as pd

import torch
import logging
import os
import random
import sys
import pandas as pd

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from scipy.special import expit
sys.path.append('')         # Set the system pth
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from models.hierbert1 import HierarchicalBert
from models.deberta import DebertaForSequenceClassification
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

device = torch.device('cuda:0')

train_dataset = pd.read_json('./data/Graph_ecthr_test.json')

length = len(train_dataset)

MODEL_PATH = './Legal-Bert/nlpaueb/legal-bert-base-uncased/ecthr_a/seed_1'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

segment_encoder = model.bert
model_encoder = HierarchicalBert(encoder=segment_encoder, max_segments=64, max_segment_length=128)
model.bert = model_encoder


model_state_dict = torch.load(f'{MODEL_PATH}/pytorch_model.bin', map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict)

model = model.to(device)
model.eval()


y = np.zeros((1000,11))
prediction = np.zeros((1000,11))

results = np.zeros((1000,64,128,11))

for i in tqdm(range(1000)):

    doc = train_dataset.loc[i,'text']
    target = np.zeros(10)
    label = train_dataset.loc[i,'labels']

    for j in range(10):
        if j in label:
            target[j]= 1
        else:
            target[j]= 0

    y_true = np.zeros(11, dtype=np.int32)
    y_true[:10] = target
    y_true[-1] = (np.sum(target) == 0).astype('int32')
    y[i,:] = y_true
    case_template = [[0] * 128]

    case_encodings = tokenizer(doc[:64], padding='max_length',max_length=128, truncation=True)

    #batch = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
    inputs = case_encodings['input_ids'] + case_template * (
            64 - len(case_encodings['input_ids']))
    attention = case_encodings['attention_mask'] + case_template * (
            64 - len(case_encodings['attention_mask']))
    tokenid = case_encodings['token_type_ids'] + case_template * (
            64 - len(case_encodings['token_type_ids']))

    inputs = np.array(inputs)
    attention = np.array(attention)
    tokenid = np.array(tokenid)

    inputs_0 = torch.IntTensor(np.array([inputs])).to(device)
    attention_0 = torch.Tensor(np.array([attention])).to(device)
    tokenid_0 = torch.IntTensor(np.array([tokenid])).to(device)

    batch = {'input_ids': inputs_0, 'attention_mask': attention_0, 'token_type_ids': tokenid_0}

    ans = model(**batch)

    logits = ans.logits.detach().cpu().numpy()

    preds = (expit(logits) > 0.5).astype('int32')
    y_pred = np.zeros(11, dtype=np.int32)
    y_pred[:10] = preds
    y_pred[-1] = (np.sum(preds) == 0).astype('int32')
    prediction[i,:] = y_pred
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)

    for row in range(64):
        if attention[row,0]==0:
            break

        for column in range(128):
            if attention[row,column]==0:
                break

            token = inputs[row,column]

            if True:

                inputs[row,column] = 103
                inputs_1 = torch.IntTensor(np.array([inputs])).to(device)
                attention_1 = torch.Tensor(np.array([attention])).to(device)
                tokenid_1 = torch.IntTensor(np.array([tokenid])).to(device)

                batch = {'input_ids': inputs_1, 'attention_mask': attention_1, 'token_type_ids': tokenid_1}

                ans = model(**batch)

                logits = ans.logits.detach().cpu().numpy()

                preds = (expit(logits) > 0.5).astype('int32')
                y_pred = np.zeros(11, dtype=np.int32)
                y_pred[:10] = preds
                y_pred[-1] = (np.sum(preds) == 0).astype('int32')
                
                results[i,row,column,:] = y_pred

                inputs[row,column] = token

np.save('./attack/Legal-Bert/ecthr_true_A.npy', y)
np.save('./attack/Legal-Bert/ecthr_pred_A.npy', prediction)
np.save('./attack/Legal-Bert/ecthr_attack_A.npy', results)
