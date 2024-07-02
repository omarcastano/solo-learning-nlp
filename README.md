# Solo Learning NLP

This repository contains implementations of well-known NLP models such as GPT and BERT. It is intended for those who want to delve deeper into NLP, and we aim to keep the implementations as simple as possible without losing important details.

## Tutorials

- Embeddings
    - [Skip-gram]() This tutorial shows how to implement and pre-train a skip-gram model using PyTorch, based on the original [paper](https://arxiv.org/abs/1301.3781) where skip-gram was proposed.

    - [CBOW]() This tutorial demonstrates how to implement and pre-train a CBOW model with PyTorch, following the original [paper](https://arxiv.org/abs/1301.3781) where CBOW was proposed.
- Neural Machine Translation (NMT)
    - [NMT with RNN]() This tutorial covers Neural Machine Translation using PyTorch and GRU recurrent neural networks. The implementation is based on the [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) paper.

    - [NMT with RNN and Attention Mechanism]() This tutorial extends Neural Machine Translation by incorporating the Bahdanau attention mechanism with GRU networks in PyTorch. It follows the paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).

    - [NMT with Transformers]() This tutorial explores Neural Machine Translation using the Transformer architecture, as detailed in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762).
- Self-Supervised:
    - [BERT]() This tutorial illustrates how to pre-train an encoder-only model using the BERT pre-training strategy, as described in the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

    - [GPT-1](https://github.com/omarcastano/solo-learning-nlp/blob/main/self-supervised/mini_gpt.ipynb) This tutorial covers the implementation of GPT-1, based on the paper [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).

    - [GPT-2]() This tutorial demonstrates the implementation of GPT-2, following the paper [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

