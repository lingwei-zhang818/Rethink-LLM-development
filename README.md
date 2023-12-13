# Rethinking the Development of Large Language Models from the Causal Perspective: A Legal Text Prediction Case Study

This repository contains the implementation of paper:<br> 
__Rethinking the Development of Large Language Models from the Causal
Perspective: A Legal Text Prediction Case Study__ <br>

This codebase is based on the implementation of [lex-glue](https://github.com/coastalcph/lex-glue)

## Prerequisites and Installation
This code is tested with python3.8.16 and Pytorch 1.11.0, transformers 4.20.1.

To run this code, follow these steps:

* Install python3.8.16 and Pytorch 1.11.0
* Install transformers 4.20.1 [Editable Version](https://huggingface.co/docs/transformers/installation)
* run: pip install -r requirements.txt


## Run

Please follow these steps:

### Data split

In this implementation, we use dataset: ECtHR-A([Chalkidis et al. (2019)](https://aclanthology.org/P19-1424/)) and ECtHR-B([Chalkidis et al. (2021a)](https://aclanthology.org/2021.naacl-main.22/))

Download the ecthr source data [here](https://zenodo.org/record/5532997/files/ecthr.tar.gz), put the `ecthr.jsonl` file to folder `./data` and run the following command.

```
python data/pre_process.py
```

### Graph Construction

Download `coref` and `openie` predictors [here](https://drive.google.com/drive/folders/1fXDI4nTqh2NV4ml0DvzmkRxhPE3duvaP?usp=share_link) to folder `./data` and run the following command:

```
python data/graph_construct.py
```

### Prepare CASAM mask

Run the following command:

```
python data/prepare_CASAM.py
```

### Train the models

Configure your system path to repository in folders: `./Legal-Bert`, `./CASAM` and `./CIESAM`.

#### Train Legal-Bert model for ECtHR A and B respectively:

```
sh scripts/base_A.sh
sh scripts/base_B.sh
```
#### Train CIESAM model:

```
sh scripts/CIESAM_A.sh
sh scripts/CIESAM_B.sh
```
#### Train CASAM model:
Replace the `modeling_bert.py` of transformers by the file in this repository and run the following command. 

Note that, We call this new environment `Env2` and the previous one `Env1`. If you want to run the previous Legal-Bert and CIESAM, put the file back and get `Env1`.

```
sh scripts/CASAM_A.sh
sh scripts/CASAM_B.sh
```


## Run attacks
After the model is trained, we can run the attack code with the saved models.

Configure your system path to repository in folders: `./attack`.
### Legal-Bert attack
Switch to `Env1` and run the following commands:
```
python attack/Legal-Bert/baseline_attack_A.py
python attack/Legal-Bert/baseline_attack_B.py
```

### CIESAM attack
Run the following commands:
```
python attack/CIESAM/CIESAM_attack_A.py
python attack/CIESAM/CIESAM_attack_B.py
```
### CASAM attack
Switch to `Env2` and run the following commands:
```
python attack/CASAM/CASAM_attack_A.py
python attack/CASAM/CASAM_attack_B.py
```
## Obtain different attack results
The model name: [Legal-Bert, CIESAM, CASAM].

The dataset: [A, B] for ECtHR-A and ECtHR-B respectively.

The attack: ['function', 'all-word', 'seq', 'dot-after-seq', 'punct', 'auxi', 'article', 'prep'] for the attacks mentioned in the paper.
```
python robustness/robustness.py --model_name Legal-Bert --dataset A --attack function
```