"""Attention-modulated Random Walk."""

import os
import csv
import pickle as pkl
import pandas as pd
import numpy as np
import random
import math
from SoftmaxLossAttn import *
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, util


def graphnli_attn_wt(node, nbd_sentences, num_candidates):
    global model, test_loss
    model.eval()

    nbd_sentences.insert(0, node)
    dataloader = DataLoader([InputExample(texts=nbd_sentences)], shuffle=False, batch_size=1)
    dataloader.collate_fn = model.smart_batching_collate

    for step, batch in enumerate(dataloader):
        features, _ = batch
        attn_weights = test_loss(features, labels=None)
    return attn_weights


def attn_modulated_graph_walk(sentences, data, node_id, child_edges, walk_len):
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
                if data['node'][c]['text'] == data['node'][c]['text']:
                    choices_text.append(data['node'][c]['text'])
                else:
                    choices_text.append('')

        if len(choices) == 0:
            return sentences, label

        attn_weights = graphnli_attn_wt(
            data['node'][node_id]['text'], choices_text, len(choices_text))

        node = choices[attn_weights.index(max(attn_weights))]
        sentences[indx] = data['node'][node]['text']
        indx += 1
        chosen_node_ids.append(node)
        node_id = node
    return sentences, label


def attn_modulated_random_walk(sentences, data, node_id, child_edges, walk_len):
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
                if data['node'][c]['text'] == data['node'][c]['text']:
                    choices_text.append(data['node'][c]['text'])
                else:
                    choices_text.append('')

        if len(choices_text) == 0:
            return sentences, label

        probs = graphnli_attn_wt(
            data['node'][node_id]['text'], choices_text, len(choices_text))

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

model_save_path = 'output/gascom_hate_attention_distilroberta-base'
model = SentenceTransformer(model_save_path)
test_loss = SoftmaxLossAttn(model=model, sentence_embedding_dimension=768, num_labels=2)


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
            sentences, label = attn_modulated_random_walk(sentences, data, node_id, child_edges, 6)

            if label != -1:
                sentences.append(label)
                dataset_samples.append(sentences)


print('#samples:', len(dataset_samples))
random.shuffle(dataset_samples)

train_samples = dataset_samples[ : math.ceil(0.8*len(dataset_samples))]
dev_samples = dataset_samples[math.ceil(0.8*len(dataset_samples)) : ]
pd.DataFrame(train_samples, columns=['sent1', 'sent2', 'sent3', 'sent4', 'sent5', 'sent6', 'label']).to_csv('./train_attn_mod_random_walk.csv', index=False)
pd.DataFrame(dev_samples, columns=['sent1', 'sent2', 'sent3', 'sent4', 'sent5', 'sent6', 'label']).to_csv('./test_attn_mod_random_walk.csv', index=False)

print('#train samples:', len(train_samples))
print('#test samples:', len(dev_samples))
