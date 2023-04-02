import torch
import torch.nn as nn
from torch.nn import functional as F

#
# This part is taken from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
# A video lecture on GPT by Andrej Karpathy
#


class SelfAttentionHead(nn.Module):
    """Single head self-attention, optionally with causal masking.
    taken from https://github.com/karpathy/ng-video-lecture,
    the explanation of nano-gpt

    :param embedding_size: the size of the input embedding
    :param sequence_len: the length of the input sequence
    :param dropout: the dropout rate
    :param head_size: the size of the attention head
    :param causal: whether to use causal masking
    """

    def __init__(
        self, embedding_size, sequence_len, dropout, head_size, causal, device
    ):
        super().__init__()
        if device is None:
            raise ValueError("Device is None at SelfAttentionHead")
        self.key = nn.Linear(embedding_size, head_size, bias=False, device=device)
        self.query = nn.Linear(embedding_size, head_size, bias=False, device=device)
        self.value = nn.Linear(embedding_size, head_size, bias=False, device=device)
        self.causal = causal
        if self.causal is True:
            self.register_buffer(
                "tril",
                torch.tril(torch.ones(sequence_len, sequence_len, device=device)),
            )
        self.dropout_val = dropout
        if self.dropout_val < 1.0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        if self.causal is True:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
            wei = F.softmax(wei, dim=-1)  # (B, T, T)
        if self.dropout_val < 1.0:
            wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel

    Note: the embedding size must be divisible by the number of heads

    :param embedding_size: the size of the input embedding
    :param sequence_len: the length of the input sequence
    :param dropout: the dropout rate
    :param num_heads: the number of attention heads
    :param head_size: the size of the attention head
    :param causal: whether to use causal masking
    """

    def __init__(
        self,
        embedding_size,
        sequence_len,
        dropout,
        num_heads,
        head_size,
        causal,
        device,
    ):
        super().__init__()
        if device is None:
            raise ValueError("Device is None at MultiHeadAttention")
        if embedding_size % num_heads != 0:
            raise ValueError(
                f"embedding_size ({embedding_size}) must be divisible by num_heads ({num_heads})"
            )
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embedding_size,
                    sequence_len,
                    dropout,
                    head_size,
                    causal,
                    device=device,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(embedding_size, embedding_size, device=device)
        self.dropout_val = dropout
        if self.dropout_val < 1.0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        if self.dropout_val < 1.0:
            out = self.dropout(self.proj(out))
        else:
            out = self.proj(out)
        return out


class FeedFoward(nn.Module):
    """Dual linear layers separated by a non-linearity

    :param input_size: the size of the input embedding
    :param hidden_size: the size of the 'hidden' linear layer in the feed-forward network,
    if None, use default input_size*4
    :param dropout: the dropout rate (default None, don't use dropout layer)
    :param non_linearity: the non-linearity to use, one of "relu" (default), "leaky_relu", "tanh"
    """

    def __init__(
        self,
        input_size,
        hidden_size=None,
        dropout=None,
        non_linearity="relu",
        device=None,
    ):
        super().__init__()
        if device is None:
            raise ValueError("Device is None at FeedFoward")
        self.device = device
        self.dropout_val = dropout
        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        elif non_linearity == "leaky_relu":
            self.non_linearity = nn.LeakyReLU()
        elif non_linearity == "tanh":
            self.non_linearity = nn.Tanh()
        if hidden_size is None or hidden_size == 0:
            hidden_size = input_size * 4
        if dropout is not None and dropout != 0:
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size, device=device),
                self.non_linearity,
                nn.Linear(hidden_size, input_size, device=device),
                nn.Dropout(dropout),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size, device=device),
                self.non_linearity,
                nn.Linear(hidden_size, input_size, device=device),
            )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation

    Note: the embedding size must be divisible by the number of heads

    :param embedding_size: the size of the input embedding
    :param sequence_len: the length of the input sequence
    :param dropout: the dropout rate
    :param num_heads: the number of attention heads
    :param causal: whether to use causal masking
    :param linear_hidden_size: the size of hidden layer in the dual-linear layer
    of the feed-forward network, if None, use embedding_size*4
    :param linear_non_linearity: the non-linearity to use in between the dual-linear layer,
    one of "relu" (default), "leaky_relu", "tanh"
    :param linear_residual: whether to use linear residual connection, default True
    """

    def __init__(
        self,
        embedding_size,
        sequence_len,
        dropout,
        num_heads,
        causal,
        linear_hidden_size=None,
        linear_non_linearity="relu",
        linear_residual=True,
        device=None,
    ):
        # embedding_size: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        if device is None:
            raise ValueError("Device is None at Block")
        head_size = embedding_size // num_heads
        self.device = device
        self.sa = MultiHeadAttention(
            embedding_size, sequence_len, dropout, num_heads, head_size, causal, device
        )
        if linear_hidden_size is None:
            linear_hidden_size = embedding_size * 4
        self.ffwd = FeedFoward(
            input_size=embedding_size,
            hidden_size=linear_hidden_size,
            dropout=dropout,
            non_linearity=linear_non_linearity,
            device=device,
        )
        self.ln1 = nn.LayerNorm(embedding_size, device=device)
        self.ln2 = nn.LayerNorm(embedding_size, device=device)
        if linear_residual is True:
            self.fRes = 1.0
        else:
            self.fRes = 0.0

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x * self.fRes + self.ffwd(self.ln2(x))
        return x


class MultiHeadSelfAttention(nn.Module):
    """MultiHeadSelfAttention transformer model (Karpathy nanoGPT derivative)

    Note: the embedding size must be divisible by the number of heads

    :param vocab_size: the size of the vocabulary
    :param embedding_size: the size of the input embedding
    :param sequence_len: the length of the input sequence
    :param dropout: the dropout rate
    :param num_heads: the number of attention heads
    :param num_layers: the number of transformer blocks
    :param causal: whether to use causal masking
    :param linear_hidden_sizes: array of number of neurons for the dual-linear layers
    respective hidden sizes (or None for default 4*embedding_size), dimension must be num_layers,
    if not None.
    :param linear_non_linearity: the non-linearity to use in between the dual-linear layers,
    :param linear_residual: whether to use linear residual connection, default True (False is only active, if hidden_sizes[i] != embedding_size*4)
    :param device: the device to use for training
    """

    def __init__(
        self,
        vocab_size,
        embedding_size,
        sequence_len,
        dropout,
        num_heads,
        num_layers,
        causal,
        linear_hidden_sizes=None,
        linear_non_linearity="relu",
        linear_residual=True,
        device=None,
    ):
        super().__init__()
        if device is None:
            raise ValueError("Device is None at MultiHeadSelfAttention")
        self.device = device
        self.sequence_len = sequence_len
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            vocab_size, embedding_size, device=device
        )
        self.position_embedding_table = nn.Embedding(
            sequence_len, embedding_size, device=device
        )
        if linear_hidden_sizes is None:
            linear_hidden_sizes = []
        if len(linear_hidden_sizes) != num_layers:
            self.blocks = nn.Sequential(
                *[
                    Block(
                        embedding_size=embedding_size,
                        sequence_len=sequence_len,
                        dropout=dropout,
                        num_heads=num_heads,
                        causal=causal,
                        linear_hidden_size=embedding_size * 4,
                        linear_non_linearity=linear_non_linearity,
                        device=device,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            blks = []
            for i in range(num_layers):
                if linear_residual is False:
                    if embedding_size * 4 != linear_hidden_sizes[i]:
                        bRes = False
                    else:
                        bRes = True
                blks.append(
                    Block(
                        embedding_size=embedding_size,
                        sequence_len=sequence_len,
                        dropout=dropout,
                        num_heads=num_heads,
                        causal=causal,
                        linear_hidden_size=linear_hidden_sizes[i],
                        linear_non_linearity=linear_non_linearity,
                        linear_residual=bRes,
                        device=device,
                    )
                )
            self.blocks = nn.Sequential(*blks)
        self.ln_f = nn.LayerNorm(embedding_size, device=device)  # final layer norm
        self.lm_head = nn.Linear(embedding_size, vocab_size, device=device)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)

        # XXX: move to init, make not trainable:
        if self.device is None:
            pos_emb = self.position_embedding_table(torch.arange(T))
        else:
            pos_emb = self.position_embedding_table(
                torch.arange(T, device=self.device)
            )  # (T,C)

        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens given a context

        Note: for apple MPS, top_k is limited max 16 vor older torchs! ((01/2023) implementation limitation)
        See: https://github.com/pytorch/pytorch/issues/78915
        Solved in: https://github.com/pytorch/pytorch/pull/94639 (03/2023)

        :param idx: the context (B,T) tensor of indices
        :param max_new_tokens: the maximum number of tokens to generate
        :param temperature: the temperature to use for sampling
        :param top_k: the number of top tokens to consider
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last sequence_len tokens
            idx_cond = idx[:, -self.sequence_len :]
            # print(idx_cond.shape)
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply temperature
            if temperature != 1.0 and temperature > 0.0:
                logits = logits / temperature
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
