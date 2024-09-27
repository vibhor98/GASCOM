"""Similarity-based Random Walk."""

import os
import csv
import pickle as pkl
import pandas as pd
import numpy as np
import random
import math
from sentence_transformers import SentenceTransformer, util


def sbert_cosine_similarity(node, nbd_sentences, num_candidates):
    global model
    nbd_sent_embeddings = model.encode(nbd_sentences, convert_to_tensor=True)
    node_embedding = model.encode(node, convert_to_tensor=True)
    hits = util.semantic_search(node_embedding, nbd_sent_embeddings, top_k=num_candidates)
    return hits[0]


def similarity_based_graph_walk(sentences, data, node_id, child_edges, walk_len):
    sentences[0] = data['node'][node_id]['text']
    label = data['node'][node_id]['label']
    chosen_node_ids = [node_id]
    indx = 1

    # Adding parent mandatorily
    edge = data['edge'][node_id]
    if len(edge.keys()) > 0:
        node_id = list(edge.keys())[0]
        if node_id not in data['node'] and node_id not in data['edge']:
            return sentences, label
        else:
            sentences[1] = data['node'][node_id]['text']
            chosen_node_ids.append(node_id)
            indx += 1
    ########

    while indx < walk_len:
        choices = []
        edge = data['edge'][node_id]
        if len(edge.keys()) > 0 and list(edge.keys())[0] not in chosen_node_ids:
            choices.append(list(edge.keys())[0])
        if node_id in child_edges:
            for child_id in child_edges[node_id]:
                if child_id not in chosen_node_ids:
                    choices.append(child_id)

        # Check for nan
        if data['node'][node_id]['text'] != data['node'][node_id]['text']:
            data['node'][node_id]['text'] = ''

        choices_text = []
        for c in choices:
            if c not in data['node'] and c not in data['edge']:
                choices.remove(c)
            else:
                if data['node'][c]['text'] != data['node'][c]['text']:
                    data['node'][c]['text'] = ''
                choices_text.append(data['node'][c]['text'])

        if len(choices) == 0:
            return sentences, label

        hit = sbert_cosine_similarity(
            data['node'][node_id]['text'], choices_text, len(choices))

        avg = math.floor(len(choices) / 2)
        node = choices[hit[avg]['corpus_id']]
        sentences[indx] = data['node'][node]['text']
        indx += 1
        chosen_node_ids.append(node)
        node_id = node
    return sentences, label


def similarity_based_random_walk(sentences, data, node_id, child_edges, walk_len):
    sentences[0] = data['node'][node_id]['text']
    label = data['node'][node_id]['label']
    chosen_node_ids = [node_id]
    indx = 1
    retries = 0

    # Adding parent mandatorily
    edge = data['edge'][node_id]
    if len(edge.keys()) > 0:
        node_id = list(edge.keys())[0]
        if node_id not in data['node'] and node_id not in data['edge']:
            return sentences, label
        else:
            sentences[1] = data['node'][node_id]['text']
            chosen_node_ids.append(node_id)
            indx += 1
    ########

    while indx < walk_len:
        choices = []
        edge = data['edge'][node_id]
        if len(edge.keys()) > 0:
            choices.append(list(edge.keys())[0])
        if node_id in child_edges:
            for child_id in child_edges[node_id]:
                choices.append(child_id)

        # Check for nan
        if data['node'][node_id]['text'] != data['node'][node_id]['text']:
            data['node'][node_id]['text'] = ''

        choices_text = []
        for c in choices:
            if c not in data['node'] and c not in data['edge']:
                choices.remove(c)
            else:
                choices_text.append(data['node'][c]['text'])

        if len(choices) == 0:
            return sentences, label

        hits = sbert_cosine_similarity(
            data['node'][node_id]['text'], choices_text, len(choices))
        probs = [abs(hit['score']) for hit in hits]

        # Normalize similarity scores to get the probability values.
        probs = probs / np.sum(probs)

        node = random.choices(choices, probs)[0]
        if node not in chosen_node_ids:
            sentences[indx] = data['node'][node]['text']
            chosen_node_ids.append(node)
            indx += 1
        else:
            retries += 1
            if retries > 4:
                break
        node_id = node
    return sentences, label


# Split dataset into train and test set.
dataset_path = './context_aware_hate/eacl_graphs/'
files = os.listdir(dataset_path)
dataset_samples = []
labels = []

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

for indx, file in enumerate(files):
    if indx >= 0:
        print('Processing', indx, file)
        data = pkl.load(open(dataset_path + file, 'rb'))

        # Just for Random Walk.
        child_edges = {}
        for node_id in data['node'].keys():
            edge = data['edge'][node_id]
            if len(edge.keys()) > 0:
                key = list(edge.keys())[0]
                if key in child_edges:
                    child_edges[key].append(node_id)
                else:
                    child_edges[key] = [node_id]

        for node_id in data['node'].keys():
            sentences = ['']*6
            sentences, label = similarity_based_random_walk(sentences, data, node_id, child_edges, 6)
            if label != -1:
                sentences.append(label)
                dataset_samples.append(sentences)


print('#samples:', len(dataset_samples))
random.shuffle(dataset_samples)

train_samples = dataset_samples[ : math.ceil(0.8*len(dataset_samples))]
dev_samples = dataset_samples[math.ceil(0.8*len(dataset_samples)) : ]
pd.DataFrame(train_samples, columns=['sent1', 'sent2', 'sent3', 'sent4', 'sent5', 'sent6', 'label']).to_csv('./train_simil_random_walk.csv', index=False)
pd.DataFrame(dev_samples, columns=['sent1', 'sent2', 'sent3', 'sent4', 'sent5', 'sent6', 'label']).to_csv('./test_simil_random_walk.csv', index=False)

print('#train samples:', len(train_samples))
print('#test samples:', len(dev_samples))
