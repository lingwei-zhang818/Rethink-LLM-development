import allennlp
from allennlp.predictors.predictor import Predictor

import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import time, math, os

from tqdm import tqdm


from spacy.lang.en import English
nlp = English()

nlp.add_pipe("sentencizer")

import networkx as nx

coref_predictor = Predictor.from_path(
    "./data/coref-spanbert.tar.gz"
)


oie_predictor = Predictor.from_path(
    "./data/openie-model.tar.gz"
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def compute_tf(word_dict, bow):
    tf_dict = {}
    bow_count = len(bow)

    for word, count in word_dict.items():
        if bow_count == 0:
            tf_dict[word] = count
        else:
            tf_dict[word] = count/float(bow_count)
    return tf_dict

def compute_idf(doc_list):
    import math
    idf_dict = {}
    N = len(doc_list)
    
    idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
    for doc in doc_list:
        for word, val in doc.items():
            if val > 0:
                idf_dict[word] += 1




    for word, val in idf_dict.items():
        idf_dict[word] = math.log10( N / float(val))
        
    return idf_dict


def compute_tfidf(tf_bow, idfs):
    tfidf = {}
    for word, val in tf_bow.items():
        tfidf[word] = val #*idfs[word]   #TODO:  remove the idfs term
    return tfidf

def compute_base_features(docs):
    # Docs are tokenized
    word_set = set(" ".join(docs).split())
    word_set_list = list(word_set)
    total_num_words = len(word_set_list)
    word_index_mapping = {}
    for idx, word in enumerate(word_set_list):
        word_index_mapping[word] = idx
    
    bow = []
    word_dict = []

    for doc in docs:
        b = doc.split()
        bow.append(b)
        wdict = dict.fromkeys(word_set, 0)
        for word in b:
            wdict[word] += 1
        word_dict.append(wdict)

    tfs_bow = []
    for i in range(len(docs)):
        tfs_bow.append(compute_tf(word_dict[i], bow[i]))

    idfs = compute_idf(word_dict)
    tfidfs = []
    for i in range(len(docs)):
        tfidfs.append(compute_tfidf(tfs_bow[i], idfs))
    return tfidfs, word_index_mapping, total_num_words

def extract_coref(coref_predictor, document):
    result = coref_predictor.predict(document=document)
    doc_tokenized = result['document']
    cluster = {}
    for c in result['clusters']:
        unique_map = " ".join(doc_tokenized[c[0][0]: c[0][1]+1])
        for ind, span in enumerate(c):
            key = "{}-{}".format(span[0], span[1])
            value = " ".join(doc_tokenized[span[0]:span[1]+1])
            cluster[key] = {"unique": unique_map, "span": value}

    return doc_tokenized, cluster

def parser_oie(result, tokens, offset):
    sample = {}
    start = None
    end = None
    key = None
    for ind,tag in enumerate(result["tags"]):
        if tag == "O":
            if start is not None:
                if end is None:
                    sample[key] = {"span": " ".join(tokens[start:start+1]), "index": [start+offset, start+offset]}
                else:
                    sample[key] = {"span": " ".join(tokens[start:end+1]), "index": [start+offset, end+offset]}
                start = None
                end = None
                key = None
        else:
            if tag[:1] == "B":
                if start is not None:
                    if end is None:
                        sample[key] = {"span": " ".join(tokens[start:start+1]), "index": [start+offset, start+offset]}
                    else:
                        sample[key] = {"span": " ".join(tokens[start:end+1]), "index": [start+offset, end+offset]}
                end = None
                start = ind
                key = tag[2:]
            else:
                end = ind
    return sample

def extract_oie(predictor, document):
    counter = 0
    ioes = []
    json_input = [{"sentence": sent.text } for sent in nlp(document).sents]
    result_list = predictor.predict_batch_json(
        json_input
    )
    for result in result_list:
        tokens = result["words"]
        for r in result["verbs"]:
            ioe_sample = parser_oie(r, tokens, counter)
            if ioe_sample.get("V") is not None and \
                ioe_sample.get("ARG0") is not None and \
                ioe_sample.get("ARG1") is not None:
                ioes.append(ioe_sample)
        counter += len(tokens)

    return ioes

def similar_match(
        text, span_list,
        text_vec=None, 
        span_vec_list=None,
        threshold=1.0
    ):

    def cosine_similarity(a, b):
        numerator = dot(a,b)
        if math.isclose(numerator, 0):
            return 0
        else:
            demoninator = norm(a) * norm(b)
            return numerator/demoninator

    match_index = -1 
    for ind, span in enumerate(span_list):
        if span==text:
            match_index=ind
            break
    if match_index != -1:
        return match_index
    elif text_vec is not None and len(span_list)!=0:
        sim_scores = []
        for vec in span_vec_list:
            sim_scores.append(cosine_similarity(vec, text_vec))
        sim_scores = np.array(sim_scores)
        max_ind = np.argmax(sim_scores)
        if sim_scores[max_ind]>threshold:
            return max_ind
        else:
            return -1
    else:
        return -1

    
def bfs_linearize(graph, root_node, directed_edges):
    visited = [root_node]
    queue = [root_node]
    linearize_graph = []
    while queue:
        s = queue.pop(0)
        ltext = []
        for n in graph.neighbors(s):
                    ltext += [f"<obj> {graph.nodes[n]['text']}"]
                    predicates = "<pred> "+" <cat> ".join([p['text'] for p in graph[s][n]["preds"]])
                    ltext += [predicates]
                
                    visited.append(n)
                    queue.append(n)

        if len(ltext) > 0:
            ltext = [f"<sub> {graph.nodes[s]['text']}"] + ltext
            linearize_graph.append(" ".join(ltext))

    return linearize_graph

def linearize_graph(graph_info):
    # Get connected subgraphs
    graph = graph_info['graph']
    directed_edges = graph_info['directed_edges']
    sub_graphs = []
    counter = 0
    for ind, c in enumerate(nx.connected_components(graph)):
        g = graph.subgraph(c)
        counter += 1
        if len(list(c)) > 1:# excluding graphs with less than 3 nodes
            sub_graphs.append(g)
    sub_graphs_sorted = sorted(sub_graphs, key=lambda x: len(x), reverse=True)
    graph_linearization = []
    for g in sub_graphs_sorted:
        g_list = list(g.nodes)
        total_nodes = len(g_list)
        root_node = g_list[np.argmax(np.array([g.nodes[i]["weight"] for i in g_list]))]
        graph_linearization += bfs_linearize(g, root_node, directed_edges)

    return graph_linearization

def construct_graph(docs, coref_predictor, oie_predictor):
    #print(docs)
    graph = nx.Graph()
    node_names_list = []
    node_vec_list = []
    node_counter = 0
    # Get Coref:
    tokenized_docs = []
    coref_info_list = []
    start_time = time.time()
    for ind, d in enumerate(docs):
        dtokens, cinfo = extract_coref(coref_predictor, d)
        tokenized_docs.append(" ".join(dtokens))
        coref_info_list.append(cinfo)
    tfidfs_list, word_index_mapping, total_num_words = compute_base_features(tokenized_docs)
    generic_tfidfs = {} # This is to handle errors when sub_name is not in current docs tfidfs
    for tfidf in tfidfs_list:
        for k,v in tfidf.items():
            if generic_tfidfs.get(k) is None:
                generic_tfidfs[k] = v
            else:
                if v>generic_tfidfs[k]:
                    generic_tfidfs[k] = v
                
    oies_list = []
    directed_edges = {}
    for ind, d in enumerate(docs):
        start_time = time.time()
        # Get OIE:
        oies = extract_oie(oie_predictor, d)
        #print(f"Time for extracting oie for doc-{ind} is {time.time()-start_time}")
        coref_info = coref_info_list[ind]
        tfidfs = tfidfs_list[ind]
        oies_list.append(oies)
        for x in oies:
            # Node for subject / ARG0
            uid = "{}-{}".format(x["ARG0"]["index"][0], x["ARG0"]["index"][1])
            if coref_info.get(uid) is not None:
                sub_name = coref_info[uid]["unique"]
            else:
                sub_name = x["ARG0"]["span"]
            # Add vector
            sub_name_vector = np.zeros(total_num_words)
            for word in sub_name.split():
                try:
                    sub_name_vector[word_index_mapping[word]]= tfidfs[word] if tfidfs.get(word) is not None else generic_tfidfs[word]
                except:
                    print(f"Exception Occurred for word: {word}")


            sim_node_ind = similar_match(sub_name, node_names_list, sub_name_vector, node_vec_list)
            if sim_node_ind == -1:
                graph.add_node(node_counter, text=sub_name, weight=1)
                sub_index = node_counter
                node_counter += 1
                node_names_list.append(sub_name)
                node_vec_list.append(sub_name_vector)
            else:
                sub_index = sim_node_ind
                graph.nodes[sub_index]["weight"] += 1


            # Node for Object / ARG1
            uid = "{}-{}".format(x["ARG1"]["index"][0], x["ARG1"]["index"][1])
            if coref_info.get(uid) is not None:
                obj_name = coref_info[uid]["unique"]
            else:
                obj_name = x["ARG1"]["span"]
            # Add vector
            obj_name_vector = np.zeros(total_num_words)
            for word in obj_name.split():
                try:
                    obj_name_vector[word_index_mapping[word]]=tfidfs[word] if tfidfs.get(word) is not None else generic_tfidfs[word]
                except:
                    print(f"Exception Occurred for word: {word}")


            sim_node_ind = similar_match(obj_name, node_names_list, obj_name_vector, node_vec_list)
            if sim_node_ind == -1:
                graph.add_node(node_counter, text=obj_name, weight=1)
                obj_index = node_counter
                node_counter += 1
                node_names_list.append(obj_name)
                node_vec_list.append(obj_name_vector)
            else:
                obj_index = sim_node_ind
                graph.nodes[obj_index]["weight"] += 1

            # Edge info
            pred_name = x["V"]["span"]
            if graph.has_edge(sub_index, obj_index):
                sim_ind = similar_match(pred_name, [x["text"] for x in graph[sub_index][obj_index]["preds"]]) # NOTE: no vector matching 
                if sim_ind==-1:
                    graph[sub_index][obj_index]["preds"].append({"text":pred_name, "weight":1})
                else:
                    graph[sub_index][obj_index]["preds"][sim_ind]["weight"] += 1
            else:
                graph.add_edge(sub_index, obj_index, preds=[{"text":pred_name, "weight":1}])
                directed_edges[f"{sub_index}-{obj_index}"] = 1

    
    extra_info = {}
    extra_info['coref_info'] = coref_info_list
    extra_info['oie_info'] = oies_list
    extra_info['tfidfs'] = tfidfs_list
    extra_info['generic_tfidfs'] = generic_tfidfs
    extra_info['word_index_mapping'] = word_index_mapping
    extra_info['total_num_words']  = total_num_words
    graph_info = {'graph': graph, 'directed_edges': directed_edges}
    return graph_info, extra_info

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


linear = []
text_new=[]
labels = []
judge_labels = []
cnt=0


df = pd.read_json("./data/ECTHR_train.json")
df['graph_text'] = df['text']

for i in tqdm(range(len(df))):


    text = df.loc[i,'text']
    
    graph = []
    new_text = []
    clock = 0
    for info in text:
        #info = 'However, it noted that as of May 2008 the first applicant had received the above-mentioned supplement for the other child in her care. She had also been entitled to the reimbursement of fees for her psychological examination, as the Călăraşi DGASPC had refused to reimburse her. In addition, she was entitled to the special allowances provided for by Article 3 of the collective agreement for the period 2007‑2009'
        graph_info, extra_info = construct_graph([info], coref_predictor, oie_predictor)
        graph_linear = linearize_graph(graph_info)
        graph_text = ' '.join(graph_linear)
        
        clock = clock+1
        if graph_text=='':
            graph.append(info)
        else:
            graph.append(graph_text)
        new_text.append(info)
        if(clock==64):
            break

    df.at[i,'graph_text'] = graph
    df.at[i, 'text'] = new_text
    
df.to_json("./data/Graph_ecthr_train.json")


df = pd.read_json("./ECTHR_test.json")
df['graph_text'] = df['text']

for i in tqdm(range(len(df))):


    text = df.loc[i,'text']
    graph = []
    new_text = []
    clock = 0
    for info in text:
        #info = 'However, it noted that as of May 2008 the first applicant had received the above-mentioned supplement for the other child in her care. She had also been entitled to the reimbursement of fees for her psychological examination, as the Călăraşi DGASPC had refused to reimburse her. In addition, she was entitled to the special allowances provided for by Article 3 of the collective agreement for the period 2007‑2009'
        graph_info, extra_info = construct_graph([info], coref_predictor, oie_predictor)
        graph_linear = linearize_graph(graph_info)
        graph_text = ' '.join(graph_linear)
        
        clock = clock+1
        if graph_text=='':
            graph.append(info)
        else:
            graph.append(graph_text)
        new_text.append(info)
        if(clock==64):
            break

    df.at[i,'graph_text'] = graph
    df.at[i, 'text'] = new_text
    
df.to_json("./data/Graph_ecthr_test.json")

df = pd.read_json("./data/ECTHR_val.json")
df['graph_text'] = df['text']

for i in tqdm(range(len(df))):


    text = df.loc[i,'text']
    #print(text)
    graph = []
    new_text = []
    clock = 0
    for info in text:
        #info = 'However, it noted that as of May 2008 the first applicant had received the above-mentioned supplement for the other child in her care. She had also been entitled to the reimbursement of fees for her psychological examination, as the Călăraşi DGASPC had refused to reimburse her. In addition, she was entitled to the special allowances provided for by Article 3 of the collective agreement for the period 2007‑2009'
        graph_info, extra_info = construct_graph([info], coref_predictor, oie_predictor)
        graph_linear = linearize_graph(graph_info)
        graph_text = ' '.join(graph_linear)
        
        clock = clock+1
        if graph_text=='':
            graph.append(info)
        else:
            graph.append(graph_text)
        new_text.append(info)
        if(clock==64):
            break

    df.at[i,'graph_text'] = graph
    df.at[i, 'text'] = new_text
    
df.to_json("./data/Graph_ecthr_val.json")