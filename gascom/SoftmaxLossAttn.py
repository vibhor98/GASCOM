import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from sentence_transformers import SentenceTransformer
import logging


logger = logging.getLogger(__name__)

class SoftmaxLossAttn(nn.Module):
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 concatenation_thesis_rep: bool = False,
                 concatenation_uv_rep: bool = False,
                 loss_fct: Callable = nn.CrossEntropyLoss()):
        super(SoftmaxLossAttn, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        self.concatenation_thesis_rep = concatenation_thesis_rep
        self.concatenation_uv_rep = concatenation_uv_rep

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        if concatenation_thesis_rep:
            num_vectors_concatenated += 1
        if concatenation_uv_rep:
            num_vectors_concatenated += 1
        logger.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep_w = ''
        if self.concatenation_uv_rep:
            rep_w = self.model_uv(sentence_features.pop())['sentence_embedding']

        reps = [self.model[0](sentence_feature)['token_embeddings'] for sentence_feature in sentence_features]

        rep_t = ''
        if self.concatenation_thesis_rep:
            rep_t = reps.pop()
        attention_weights = self.multihead_attention(reps)
        return attention_weights

    def multihead_attention(self, reps):
        u = []
        for i in range(1, len(reps)):
            Q = self.model[3](reps[1])
            K = self.model[4](reps[i])
            V = self.model[5](reps[i])
            attn_output, attn_weights = self.model[1](Q, K, V)
            u.append(attn_weights.detach().numpy())
        return u
