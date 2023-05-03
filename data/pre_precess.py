import pandas as pd
from datasets import load_dataset

a = pd.read_json('./data/ecthr.jsonl',lines=True)
labels = ['2', '3', '5', '6', '8', '9', '10', '11', '14', 'P1-1']

train=[]
val = []
test =[]

for i in range(len(a)):
    x = a.iloc[i,:]

    if x['data_type']=='train':
        tmp = x['allegedly_violated_articles']
        label = []
        for items in tmp:
            if items in labels:
                label.append(labels.index(items))
        x['allegedly_violated_articles'] = label

        tmp = x['violated_articles']
        label = []
        for items in tmp:
            if items in labels:
                label.append(labels.index(items))
        x['violated_articles'] = label
        train.append(x)

    elif x['data_type']=='test':
        tmp = x['allegedly_violated_articles']
        label = []
        for items in tmp:
            if items in labels:
                label.append(labels.index(items))
        x['allegedly_violated_articles'] = label

        tmp = x['violated_articles']
        label = []
        for items in tmp:
            if items in labels:
                label.append(labels.index(items))
        x['violated_articles'] = label
        test.append(x)
    else:
        tmp = x['allegedly_violated_articles']
        label = []
        for items in tmp:
            if items in labels:
                label.append(labels.index(items))
        x['allegedly_violated_articles'] = label

        tmp = x['violated_articles']
        label = []
        for items in tmp:
            if items in labels:
                label.append(labels.index(items))
        x['violated_articles'] = label
        val.append(x)



df_train = pd.DataFrame(train)
df_test = pd.DataFrame(test)
df_val = pd.DataFrame(val)



df_train = df_train.drop(columns=['case_id', 'case_no', 'data_type', 'title', 'gold_rationales', 'judgment_date', 'applicants', 'defendants', 'court_assessment_references', 'silver_rationales'])
df_train = df_train.rename(columns={'facts': 'text', 'allegedly_violated_articles': "judge_labels", 'violated_articles': "labels"})
df_train = df_train.reset_index()
df_train.to_json('./data/ECTHR_train.json')

df_val = df_val.drop(columns=['case_id', 'case_no', 'data_type', 'title', 'gold_rationales', 'judgment_date', 'applicants', 'defendants', 'court_assessment_references', 'silver_rationales'])
df_val = df_val.rename(columns={'facts': 'text', 'allegedly_violated_articles': "judge_labels", 'violated_articles': "labels"})
df_val = df_val.reset_index()
df_val.to_json('./data/ECTHR_val.json')

df_test = df_test.drop(columns=['case_id', 'case_no', 'data_type', 'title', 'gold_rationales', 'judgment_date', 'applicants', 'defendants', 'court_assessment_references', 'silver_rationales'])
df_test = df_test.rename(columns={'facts': 'text', 'allegedly_violated_articles': "judge_labels", 'violated_articles': "labels"})
df_test = df_test.reset_index()
df_test.to_json('./data/ECTHR_test.json')
