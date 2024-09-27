"""Latest code for GASCOM model."""

import math
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import os
import gzip
import csv
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from torch.nn import Linear, MultiheadAttention
from torch.utils.data import DataLoader

from LabelAccuracyEvaluator import *
from SoftmaxLoss import *
from layers import Dense, MultiHeadAttention


model_name = sys.argv[1] if len(sys.argv) > 1 else 'distilroberta-base'

train_batch_size = 8


model_save_path = 'output/gascom_hate_attention_' + model_name.replace("/", "-")

word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

dense_model = Dense.Dense(in_features=3*760, out_features=2)
multihead_attn = MultiHeadAttention.MultiHeadAttention(760, 5, batch_first=True)

linear_proj_q = Dense.Dense(word_embedding_model.get_word_embedding_dimension(), 760)
linear_proj_k = Dense.Dense(word_embedding_model.get_word_embedding_dimension(), 760)
linear_proj_v = Dense.Dense(word_embedding_model.get_word_embedding_dimension(), 760)
linear_proj_node = Dense.Dense(word_embedding_model.get_word_embedding_dimension(), 760)

model = SentenceTransformer(modules=[word_embedding_model, multihead_attn, dense_model, linear_proj_q, linear_proj_k, linear_proj_v, linear_proj_node])
model_uv = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_samples = []
test_samples = []

trainset = pd.read_csv('train_simil_random_walk_parent.csv')
trainset = trainset.fillna('')

for i in range(len(trainset)):
    texts = []
    for j in range(1, 7):
        texts.append(trainset.iloc[i]['sent' + str(j)])
    train_samples.append(InputExample(texts=texts, label=int(trainset.iloc[i]['label'])))

dev_samples = train_samples[math.ceil(0.8 * len(train_samples)) : ]
train_samples = train_samples[ : math.ceil(0.8 * len(train_samples)) ]

testset = pd.read_csv('./test_simil_random_walk_parent.csv')
testset = testset.fillna('')

for i in range(len(testset)):
    texts = []
    for j in range(1, 7):
        texts.append(testset.iloc[i]['sent' + str(j)])
    test_samples.append(InputExample(texts=texts, label=int(testset.iloc[i]['label'])))

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
dev_dataloader = DataLoader(dev_samples, shuffle=True, batch_size=train_batch_size)
test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=train_batch_size)

for i in range(3):
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = SoftmaxLoss(model=model, model_uv=model_uv, multihead_attn=multihead_attn, linear_proj_q=linear_proj_q,
        linear_proj_k=linear_proj_k, linear_proj_v=linear_proj_v, linear_proj_node=linear_proj_node,
        sentence_embedding_dimension=pooling_model.get_sentence_embedding_dimension(),
        num_labels=2)

    dev_evaluator = LabelAccuracyEvaluator(dev_dataloader, name='sts-dev', softmax_model=train_loss)

    num_epochs = 3
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=num_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=model_save_path
    )

test_evaluator = LabelAccuracyEvaluator(test_dataloader, name='sts-test', softmax_model=train_loss)
test_evaluator(model, output_path=model_save_path)
