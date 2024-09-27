import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging


logger = logging.getLogger(__name__)

class SoftmaxLoss(nn.Module):
    """
    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.
    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?
    :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss()
    """
    def __init__(self,
                 model: SentenceTransformer,
                 model_uv: SentenceTransformer | None,
                 multihead_attn,
                 linear_proj_q,
                 linear_proj_k,
                 linear_proj_v,
                 linear_proj_node,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 concatenation_thesis_rep: bool = False,
                 concatenation_uv_rep: bool = False,
                 loss_fct: Callable = nn.CrossEntropyLoss()):
        super(SoftmaxLoss, self).__init__()
        self.model = model
        self.model_uv = model_uv
        self.multihead_attn = multihead_attn
        self.linear_proj_q = linear_proj_q
        self.linear_proj_k = linear_proj_k
        self.linear_proj_v = linear_proj_v
        self.linear_proj_node = linear_proj_node
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        self.concatenation_thesis_rep = concatenation_thesis_rep
        self.concatenation_uv_rep = concatenation_uv_rep

        self.model.eval()

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
        self.model.eval()

        rep_w = ''
        if self.concatenation_uv_rep:
            rep_w = self.model_uv(sentence_features.pop())['sentence_embedding']

        reps = [self.model[0](sentence_feature)['token_embeddings'] for sentence_feature in sentence_features]

        rep_t = ''
        if self.concatenation_thesis_rep:
            rep_t = reps.pop()

        rep_a, rep_b = self.multihead_attention(reps)
        # rep_a, rep_b = self.cosine_sim_aggregate(reps)

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        if self.concatenation_thesis_rep:
            vectors_concat.append(rep_t)

        if self.concatenation_uv_rep:
            vectors_concat.append(rep_w)

        features = torch.cat(vectors_concat, 1)

        output = self.model[2](features)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output

    def multihead_attention(self, reps):
        v = self.model[6](reps[0])
        v = self.mean_pooling(v)
        u = []
        for i in range(1, 6):
            Q = self.model[3](reps[1])
            K = self.model[4](reps[i])
            V = self.model[5](reps[i])
            attn_output = self.model[1](Q, K, V)[0]
            u.append(self.mean_pooling(attn_output))
        u = torch.mean(torch.stack(u), dim=0)
        return u, v

    def cosine_sim_aggregate(self, reps):
        v = reps[0]
        for i in range(2, 6):
            weight = cosine_similarity(Tensor.cpu(reps[1]).detach().numpy(), Tensor.cpu(reps[i]).detach().numpy())
            reps[i] = torch.mm(torch.from_numpy(weight).to('cuda:0'), reps[i])
        u = torch.sum(torch.stack(reps[1:]), dim=0)
        return u, v
