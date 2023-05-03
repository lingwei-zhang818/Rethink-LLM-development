import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from transformers import (
    AutoTokenizer,
)
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Model Description')
parser.add_argument('--model_name', type=str, help= 'Select model', default='Legal-Bert')
parser.add_argument('--dataset', type=str, help='Select Ecthr A or B', choices={'A', 'B'}, default='A')
parser.add_argument('--attack', type=str, help='Select attack mode', choices={'function', 'all-word', 'seq', 'dot-after-seq', 'punct', 'auxi', 'article', 'prep'}, default='function')
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
PTH=f"./attack/{args.model_name}/"

y_true = np.load(PTH+f'ecthr_true_{args.dataset}.npy', allow_pickle=True)
mask_pred = np.load(PTH+f'ecthr_pred_{args.dataset}.npy', allow_pickle=True)
mask_attack = np.load(PTH+f'ecthr_attack_{args.dataset}.npy', allow_pickle=True)

dataset = pd.read_json('./data/Graph_ecthr_mask_test.json')

flag = 0
attack_name = ['function', 'auxi', 'prep', 'article', 'punct']
confounder_indexes = np.load('./robustness/indexes.npy', allow_pickle=True).item()
if args.attack in attack_name:
    confounder_index = confounder_indexes[args.attack]
elif args.attack == 'all-word':
    flag = 1
    confounder_index = confounder_indexes['article']
else:
    flag = 2

y_pred = mask_pred
attack = mask_attack

pos = 0
num = 0


y_attack = y_pred.copy()

micro_f1 = f1_score(y_true=y_true, y_pred=y_pred,average='micro', zero_division=0)
macro_f1 = f1_score(y_true=y_true, y_pred=y_pred,average='macro', zero_division=0)
print(micro_f1, macro_f1)

 
for i in tqdm(range(1000)):

    doc = dataset.loc[i,'text']
    case_encodings = tokenizer(doc[:64], padding='max_length',max_length=128, truncation=True)
    case_template = [[0] * 128]

    inputs = case_encodings['input_ids'] + case_template * (
            64 - len(case_encodings['input_ids']))
    attention = case_encodings['attention_mask'] + case_template * (
            64 - len(case_encodings['attention_mask']))
    inputs = np.array(inputs)
    attention = np.array(attention)

    min_f1 = 1.0

    for row in range(64):
        if attention[row,0]==0:
            break


        if flag == 2:
            if args.attack == 'seq':
                column = 1
            else:
                column = 2

            if True:
                cmp_f1 = f1_score(y_true=y_pred[i,:], y_pred=attack[i,row,column,:],average='micro', zero_division=0)

                if cmp_f1 < min_f1:
                    min_f1 = cmp_f1

                    y_attack[i,:] = attack[i,row,column,:]
                if cmp_f1 == 1.0:
                    pos = pos + 1

                num = num + 1
            continue

        for column in range(128):
            if attention[row,column] == 0:
                break
            token = inputs[row,column]
            if (token in confounder_index and flag == 0) or flag == 1:
                cmp_f1 = f1_score(y_true=y_pred[i,:], y_pred=attack[i,row,column,:],average='micro', zero_division=0)

                if cmp_f1 < min_f1:
                    min_f1 = cmp_f1

                    y_attack[i,:] = attack[i,row,column,:]
                if cmp_f1 == 1.0:
                    pos = pos + 1

                num = num + 1

micro_f1 = f1_score(y_true=y_true, y_pred=y_attack,average='micro', zero_division=0)
macro_f1 = f1_score(y_true=y_true, y_pred=y_attack,average='macro', zero_division=0)

print(f"Micro_f1: {micro_f1}, Macro_f1: {macro_f1}")
print(f"CR: {pos/num}")