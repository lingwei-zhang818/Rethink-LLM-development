import pandas as pd
import transformers
from transformers import (
    AutoTokenizer,
)
import numpy as np
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

data = pd.read_json('./data/Graph_ecthr_train.json')

data['mask'] = data['text']

length = len(data)

for i in tqdm(range(length)):
    mask = np.zeros((64,128))
    doc = data.loc[i,'text']
    graph=data.loc[i,'graph_text']
    for row in range(len(doc)):
        A = tokenizer(doc[row], padding="max_length", max_length=128, truncation=True)
        token_text = A['input_ids']
        B = tokenizer(graph[row], padding="max_length", max_length=128, truncation=True)
        token_graph = B['input_ids']
        for column in range(128):
            mark = token_text[column]
            if mark == 0:
                break
            if mark == 101 or mark == 102:
                continue
            if mark in token_graph:
                mask[row,column]=1
    mask = mask.tolist()
    data.at[i,'mask'] = mask
data.to_json('./data/Graph_ecthr_mask_train.json')

data = pd.read_json('./data/Graph_ecthr_test.json')

data['mask'] = data['text']

length = len(data)

for i in tqdm(range(length)):
    mask = np.zeros((64,128))
    doc = data.loc[i,'text']
    graph=data.loc[i,'graph_text']
    for row in range(len(doc)):
        A = tokenizer(doc[row], padding="max_length", max_length=128, truncation=True)
        token_text = A['input_ids']
        B = tokenizer(graph[row], padding="max_length", max_length=128, truncation=True)
        token_graph = B['input_ids']
        for column in range(128):
            mark = token_text[column]
            if mark == 0:
                break
            if mark == 101 or mark == 102:
                continue
            if mark in token_graph:
                mask[row,column]=1
    mask = mask.tolist()
    data.at[i,'mask'] = mask
data.to_json('./data/Graph_ecthr_mask_test.json')

data = pd.read_json('./data/Graph_ecthr_val.json')

data['mask'] = data['text']

length = len(data)

for i in tqdm(range(length)):
    mask = np.zeros((64,128))
    doc = data.loc[i,'text']
    graph=data.loc[i,'graph_text']
    for row in range(len(doc)):
        A = tokenizer(doc[row], padding="max_length", max_length=128, truncation=True)
        token_text = A['input_ids']
        B = tokenizer(graph[row], padding="max_length", max_length=128, truncation=True)
        token_graph = B['input_ids']
        for column in range(128):
            mark = token_text[column]
            if mark == 0:
                break
            if mark == 101 or mark == 102:
                continue
            if mark in token_graph:
                mask[row,column]=1
    mask = mask.tolist()
    data.at[i,'mask'] = mask
data.to_json('./data/Graph_ecthr_mask_val.json')