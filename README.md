# GASCOM: Graph-based Attentive Semantic Context Modeling for Online Conversation Understanding

**Vibhor Agarwal**, Yu Chen, Nishanth Sastry, "GASCOM: Graph-based Attentive Semantic Context Modeling for Online Conversation Understanding", Online Social Networks aqnd Media (OSNEM) journal, 2024.

## Abstract

Online conversation understanding is an important yet challenging NLP problem which has many useful applications (e.g., hate speech detection). However, online conversations typically unfold over a series of posts and replies to those posts, forming a tree structure within which individual posts may refer to semantic context from higher up the tree. Such semantic cross-referencing makes it difficult to understand a single post by itself; yet considering the entire conversation tree is not only difficult to scale but can also be misleading as a single conversation may have several distinct threads or points, not all of which are relevant to the post being considered. In this paper, we propose a **G**raph-based **A**ttentive **S**emantic **CO**ntext **M**odeling (GASCOM) framework for online conversation understanding. Specifically, we design two novel algorithms that utilise both the graph structure of the online conversation as well as the semantic information from individual posts for retrieving relevant context nodes from the whole conversation. We further design a token-level multi-head graph attention mechanism to pay different attentions to different tokens from different selected context utterances for fine-grained conversation context modeling. Using this semantic conversa- tional context, we re-examine two well-studied problems: polarity prediction and hate speech detection. Our proposed framework significantly outperforms state-of-the-art methods on both tasks, improving macro-F1 scores by 4.5% for polarity prediction and by 5% for hate speech detection. The GASCOM context weights also enhance interpretability.

The paper PDF is available [here](https://arxiv.org/abs/2310.14028).


## Citation

If you find this paper useful in your research, please consider citing:

```
@article{agarwal2023gascom,
  title={GASCOM: Graph-based Attentive Semantic Context Modeling for Online Conversation Understanding},
  author={Agarwal, Vibhor and Chen, Yu and Sastry, Nishanth},
  journal={arXiv preprint arXiv:2310.14028},
  year={2023}
}
```
