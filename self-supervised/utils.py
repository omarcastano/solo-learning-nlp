import re
from typing import List, Union
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import string


# Basic preprocessing.
def text_preprocessing(text: str) -> str:

    # to lower
    text = text.lower()

    # remove number
    text = re.sub(r"\d+", "", text)

    # remove html tags
    text = re.sub(r"<.*?>", "", text)

    # remove scape sequences (\n , \t, ...)
    # text = re.sub(r"\\n|\\t|\\r", "", text)

    # remove all punctuation marks except commas and dots
    keep = {',', '.', "'"}
    remove = set(string.punctuation) - keep
    translation_table = str.maketrans('', '', ''.join(remove))
    text.translate(translation_table)

    # remove urls
    text = re.sub(r"https?://\S+", "", text)

    # remove white spaces at the start and the end
    text = text.strip()

    return text

# Define embedding class using positional and token embeddings
class Embedding(nn.Module):
    """
    Embedding class for BERT-like models.

    Arguments:
    ----------
        vocab_size: int
            Size of the vocabulary.
        embed_dim: int
            Size of the embedding dimension.
        max_seq_len: int
            Maximum sequence length.
    """

    def __init__(self, vocab_size, embed_dim, max_seq_len, device="cpu"):

        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim, device=device)
        self.positional_encodings = nn.Embedding(max_seq_len, embed_dim, device=device)
        self.device = device

    def forward(self, x):
        

        token_embeddings = self.token_embeddings(x)

        x_pos = torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1).to(self.device)

        positional_encodings = self.positional_encodings(x_pos)

        embedding = token_embeddings + positional_encodings

        return embedding


# define MultiHeadAttention
class MultiHeadAttention(nn.Module):
    """
    Thi class implements MultiHeadAttention.

    Arguments:
    ----------
        embed_dim: int
            The dimension of the embedding.
        num_heads: int
            The number of heads.
        dropout: float
            The dropout rate.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0) -> None:

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_head = num_heads
        self.head_dim = embed_dim // num_heads

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q shape: (batch_size, seq_len, embed_dim)
        # k shape: (batch_size, seq_len, embed_dim)
        # v shape: (batch_size, seq_len, embed_dim)

        batch_size = q.shape[0]
        seq_len = q.shape[1]

        Q = self.w_q(q)  # Q shape : (batch_size, seq_len, embed_dim)
        K = self.w_k(k)  # K shape : (batch_size, seq_len, embed_dim)
        V = self.w_v(v)  # V shape : (batch_size, seq_len, embed_dim)

        # Q shape: (batch_size, num_head, seq_len, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        # K shape: (batch_size, num_head, seq_len, head_dim)
        K = K.reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        # V shape: (batch_size, num_head, seq_len, head_dim)
        V = V.reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(self.head_dim)  # energy shape: (batch_size, num_head, seq_len, seq_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -float("inf"))

        attention = torch.matmul(F.softmax(energy, dim=-1), V)  # attention shape: (batch_size, num_head, seq_len, head_dim)

        attention = self.dropout(attention)

        Z = self.w_o(
            attention.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.num_head * self.head_dim)
        )  # Z shape: (batch_size, seq_len, embed_dim)

        return Z, attention


# define add and normalize layer
class AddAndNormalize(nn.Module):
    """
    This class implements the add and normalize layer.

    Arguments:
    ----------
        embed_dim: int
            The dimension of the embeddings.
    """

    def __init__(self, embed_dim) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, z):
        return self.layer_norm(x + z)


# define PositionWise FFN class
class PositionWiseFFN(nn.Module):
    """
    This class implements the position wise feed forward network.

    Arguments:
    ----------
        embed_dim: int
            The dimension of the input.
        pf_dim: int
            The dimension of the hidden layer.
        dropout: float
            The dropout rate.
    """

    def __init__(self, embed_dim, pf_dim, dropout) -> None:
        super().__init__()

        self.w1 = nn.Linear(embed_dim, pf_dim)
        self.w2 = nn.Linear(pf_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):

        x = self.w1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.w2(x)

        return x


# define Encoder Layer class
class EncoderLayer(nn.Module):
    """
    This class defines the Encoder Layer.
    It consists of a multi-head attention layer and a feed forward layer.

    Arguments:
    ----------
        embed_dim: int
            The dimension of the embedding vector.
        num_heads: int
            The number of attention heads.
        dropout: float
            The dropout rate.
        pf_dim: int
            The dimension of the feed forward layer.
    """

    def __init__(self, embed_dim, num_heads, dropout, pf_dim):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feedforward = PositionWiseFFN(embed_dim, pf_dim, dropout)
        self.add_and_norm_1 = AddAndNormalize(embed_dim)
        self.add_and_norm_2 = AddAndNormalize(embed_dim)

    def forward(self, x, mask):

        # x shape: (batch_size, seq_len, embed_dim)
        # mask shape: (batch_size, 1, 1, seq_len)

        z, _ = self.multihead_attention(x, x, x, mask)
        x = self.add_and_norm_1(x, z)

        z = self.feedforward(x)
        x = self.add_and_norm_2(x, z)

        return x


# Define the EncoderTransformer layer
class EncoderTransformer(nn.Module):
    """
    This class implements the EncoderTransformer layer, which is a stack of EncoderLayer layers.

    Arguments:
    ----------
        embed_dim: int
            The dimension of the embedding layer.
        num_heads: int
            The number of attention heads.
        dropout: float
            The dropout rate.
        pf_dim: int
            The dimension of the feedforward layer.
        vocab_size: int
            The size of the vocabulary.
        max_seq_length: int
            The maximum sequence length.
        n_layers: int
            The number of EncoderLayer layers.
        device: str
            The device to use.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        pf_dim,
        vocab_size,
        max_seq_length,
        n_layers,
        device="cpu",
    ) -> None:
        super().__init__()

        self.embedding = Embedding(vocab_size, embed_dim, max_seq_length, device)
        self.encoder = nn.ModuleList([EncoderLayer(embed_dim, num_heads, dropout, pf_dim) for _ in range(n_layers)])
        self.to(device)

    def forward(self, x, mask):

        # x shape: (batch_size, seq_length)
        # mask shape: (batch_size, 1, 1, seq_length)

        x = self.embedding(x)  # x shape:  (batch_size, seq_length, embed_dim)

        for layer in self.encoder:
            x = layer(x, mask)  # x shape: (batch_size, seq_length, embed_dim)

        return x

# Define modified decoder layer
class ModifiedDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, pf_dim, dropout):
        super().__init__()

        self.mask_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feedforward = PositionWiseFFN(embed_dim, pf_dim, dropout)

        self.add_and_norm_1 = AddAndNormalize(embed_dim)
        self.add_and_norm_2 = AddAndNormalize(embed_dim)

    def forward(self, x, decoder_mask):
        # x shape: (batch_size, seq_len, embed_dim)
        # decoder_mask shape: (batch_size, 1, seq_len, seq_len)

        z, _ = self.mask_attention(x, x, x, mask=decoder_mask)
        x = self.add_and_norm_1(x, z)

        z = self.feedforward(x)
        x = self.add_and_norm_2(x, z)

        return x
    

# Define the DecoderTransformer layer
class DecoderTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, pf_dim, vocab_size, max_seq_length, n_layers, device="cpu") -> None:
        super().__init__()

        self.embedding = Embedding(vocab_size, embed_dim, max_seq_length, device)
        self.decoder = nn.ModuleList([ModifiedDecoderLayer(embed_dim, num_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.to(device)

    def forward(self, x, decoder_mask):
        x = self.embedding(x)

        for layer in self.decoder:
            x = layer(x, decoder_mask)

        return x