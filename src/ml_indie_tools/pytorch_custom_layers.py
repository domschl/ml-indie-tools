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

    def __init__(self, embedding_size, sequence_len, dropout, head_size, causal):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.causal = causal
        if self.causal is True:
            self.register_buffer(
                "tril", torch.tril(torch.ones(sequence_len, sequence_len))
            )
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
        self, embedding_size, sequence_len, dropout, num_heads, head_size, causal
    ):
        super().__init__()
        if embedding_size % num_heads != 0:
            raise ValueError(
                f"embedding_size ({embedding_size}) must be divisible by num_heads ({num_heads})"
            )
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embedding_size, sequence_len, dropout, head_size, causal
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """Simple linear layer followed by a non-linearity

    :param embedding_size: the size of the input embedding
    :param dropout: the dropout rate
    """

    def __init__(self, embedding_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout),
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
    """

    def __init__(self, embedding_size, sequence_len, dropout, num_heads, causal):
        # embedding_size: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        head_size = embedding_size // num_heads
        self.sa = MultiHeadAttention(
            embedding_size, sequence_len, dropout, num_heads, head_size, causal
        )
        self.ffwd = FeedFoward(embedding_size, dropout)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
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
        device,
    ):
        self.device = device
        self.sequence_len = sequence_len
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(sequence_len, embedding_size)
        self.blocks = nn.Sequential(
            *[
                Block(
                    embedding_size,
                    sequence_len=sequence_len,
                    dropout=dropout,
                    num_heads=num_heads,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embedding_size)  # final layer norm
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)

        # XXX: move to init, make not trainable:
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

    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last sequence_len tokens
            idx_cond = idx[:, -self.sequence_len :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
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
