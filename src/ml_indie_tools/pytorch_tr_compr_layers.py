import torch
import torch.nn as nn
from torch.nn import functional as F

from ml_indie_tools.pytorch_custom_layers import MultiHeadAttention

#
# Many parts are taken from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
# A video lecture on GPT by Andrej Karpathy
#


class FeedForwardWithCompression(nn.Module):
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
            raise ValueError("Device None at FeedForwardWithCompression")
        self.dropout_val = dropout
        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        elif non_linearity == "leaky_relu":
            self.non_linearity = nn.LeakyReLU()
        elif non_linearity == "tanh":
            self.non_linearity = nn.Tanh()
        else:
            self.non_linearity = nn.ReLU()
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


class FeedForwardWithCompressionState(nn.Module):
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
            raise ValueError("Device None at FeedForwardWithCompressionState")
        self.dropout_val = dropout
        self.net1 = nn.Linear(input_size, hidden_size, device=device)
        if non_linearity == "relu":
            self.net2 = nn.ReLU()
            # self.gateact = nn.Identity()
        elif non_linearity == "leaky_relu":
            self.net2 = nn.LeakyReLU()
            # self.gateact = nn.Identity()
        elif non_linearity == "tanh":
            self.net2 = nn.Tanh()
            # self.gateact = nn.Identity()
        elif non_linearity == "relurelu":
            self.net2 = nn.ReLU()
            # self.gateact = nn.ReLU()
        elif non_linearity == "leaky_relurelu":
            self.net2 = nn.LeakyReLU()
            # self.gateact = nn.LeakyReLU()
        elif non_linearity == "tanhhanh":
            self.net2 = nn.Tanh()
            # self.gateact = nn.Tanh()
        else:
            self.net2 = nn.ReLU()
            # self.gateact = nn.Identity()
        if hidden_size is None or hidden_size == 0:
            hidden_size = input_size * 4
        self.net3 = nn.Linear(hidden_size, input_size, device=device)
        if dropout is not None and dropout != 0:
            self.net4 = nn.Dropout(dropout)
        else:
            self.net4 = nn.Identity()
        self.gatenet = nn.Linear(hidden_size, hidden_size, device=device)
        self.state_zero = torch.zeros((1, input_size, hidden_size), device=device)

    def forward(self, x, state):
        x = self.net1(x)
        state = self.gatenet(self.state_zero[:, -x.shape[1] :, :] + state)
        # state = self.gateact(state)
        x = x + state
        x = self.net2(x)
        state = x
        x = self.net3(x)
        x = self.net4(x)
        return x, state


class BlockWithCompression(nn.Module):
    """Transformer block: communication followed by computation

    Note: the embedding size must be divisible by the number of heads

    :param embedding_size: the size of the input embedding
    :param sequence_len: the length of the input sequence
    :param dropout: the dropout rate
    :param num_heads: the number of attention heads
    :param causal: whether to use causal masking
    :param linear_non_linearity: the non-linearity to use in between the dual-linear layer,
    one of "relu" (default), "leaky_relu", "tanh"
    :param linear_hidden_size: the size of the 'hidden' linear layer in the feed-forward network, None for default 4*embedding_size
    """

    def __init__(
        self,
        embedding_size,
        sequence_len,
        dropout,
        num_heads,
        causal,
        linear_non_linearity="relu",
        linear_hidden_size=None,
        device=None,
    ):
        # embedding_size: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        if device is None:
            raise ValueError("Device is None at BlockWithCompression")
        self.device = device
        head_size = embedding_size // num_heads
        self.sa = MultiHeadAttention(
            embedding_size, sequence_len, dropout, num_heads, head_size, causal, device
        )
        if linear_hidden_size is None:
            linear_hidden_size = embedding_size * 4

        self.ffwd = FeedForwardWithCompression(
            input_size=embedding_size,
            hidden_size=linear_hidden_size,
            dropout=dropout,
            non_linearity=linear_non_linearity,
            device=device,
        )
        self.ln1 = nn.LayerNorm(embedding_size, device=device)
        self.ln2 = nn.LayerNorm(embedding_size, device=device)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BlockWithCompressionState(nn.Module):
    """Transformer block: communication followed by computation

    Note: the embedding size must be divisible by the number of heads

    :param embedding_size: the size of the input embedding
    :param sequence_len: the length of the input sequence
    :param dropout: the dropout rate
    :param num_heads: the number of attention heads
    :param causal: whether to use causal masking
    :param linear_non_linearity: the non-linearity to use in between the dual-linear layer,
    one of "relu", "leaky_relu", "tanh" (default)
    :param linear_hidden_size: the size of the 'hidden' linear layer in the feed-forward network, None for default 4*embedding_size
    """

    def __init__(
        self,
        embedding_size,
        sequence_len,
        dropout,
        num_heads,
        causal,
        linear_non_linearity="tanh",
        linear_hidden_size=None,
        device=None,
    ):
        # embedding_size: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        if device is None:
            raise ValueError("Device is None at BlockWithCompressionState.")
        self.device = device
        head_size = embedding_size // num_heads
        self.sa = MultiHeadAttention(
            embedding_size, sequence_len, dropout, num_heads, head_size, causal, device
        )
        if linear_hidden_size is None:
            linear_hidden_size = embedding_size * 4
        self.ffwd = FeedForwardWithCompressionState(
            input_size=embedding_size,
            hidden_size=linear_hidden_size,
            dropout=dropout,
            non_linearity=linear_non_linearity,
            device=device,
        )
        self.ln1 = nn.LayerNorm(embedding_size, device=device)
        self.ln2 = nn.LayerNorm(embedding_size, device=device)

    def forward(self, x, state):
        # x = x.to(self.device)
        # state = state.to(self.device)
        x = x + self.sa(self.ln1(x))
        y, state = self.ffwd(self.ln2(x), state)
        x = x + y
        return x, state


class BlockWithCompressionNoYokeResidual(nn.Module):
    """Transformer block: communication followed by computation

    Note: the embedding size must be divisible by the number of heads

    :param embedding_size: the size of the input embedding
    :param sequence_len: the length of the input sequence
    :param dropout: the dropout rate
    :param num_heads: the number of attention heads
    :param causal: whether to use causal masking
    :param linear_non_linearity: the non-linearity to use in between the dual-linear layer,
    one of "relu" (default), "leaky_relu", "tanh"
    :param linear_hidden_size: the size of the 'hidden' linear layer in the feed-forward network, None for default 4*embedding_size
    """

    def __init__(
        self,
        embedding_size,
        sequence_len,
        dropout,
        num_heads,
        causal,
        linear_non_linearity="relu",
        linear_hidden_size=None,
        device=None,
    ):
        # embedding_size: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        if device is None:
            raise ValueError("Device is None at BlockWithCompressionNoYokeResidual.")

        head_size = embedding_size // num_heads
        self.sa = MultiHeadAttention(
            embedding_size, sequence_len, dropout, num_heads, head_size, causal, device
        )
        if linear_hidden_size is None:
            linear_hidden_size = embedding_size * 4
            self.use_residual = True
        else:
            self.use_residual = False

        self.ffwd = FeedForwardWithCompression(
            input_size=embedding_size,
            hidden_size=linear_hidden_size,
            dropout=dropout,
            non_linearity=linear_non_linearity,
            device=device,
        )
        self.ln1 = nn.LayerNorm(embedding_size, device=device)
        self.ln2 = nn.LayerNorm(embedding_size, device=device)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = self.ffwd(self.ln2(x))
        return x


class BlockWithCompressionStateNoYokeResidual(nn.Module):
    """Transformer block: communication followed by computation

    Note: the embedding size must be divisible by the number of heads

    :param embedding_size: the size of the input embedding
    :param sequence_len: the length of the input sequence
    :param dropout: the dropout rate
    :param num_heads: the number of attention heads
    :param causal: whether to use causal masking
    :param linear_non_linearity: the non-linearity to use in between the dual-linear layer,
    one of "relu", "leaky_relu", "tanh" (default)
    :param linear_hidden_size: the size of the 'hidden' linear layer in the feed-forward network, None for default 4*embedding_size
    """

    def __init__(
        self,
        embedding_size,
        sequence_len,
        dropout,
        num_heads,
        causal,
        linear_non_linearity="tanh",
        linear_hidden_size=None,
        device=None,
    ):
        # embedding_size: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        if device is None:
            raise ValueError(
                "Device is None at BlockWithCompressionStateNoYokeResidual."
            )

        self.device = device
        head_size = embedding_size // num_heads
        self.sa = MultiHeadAttention(
            embedding_size, sequence_len, dropout, num_heads, head_size, causal, device
        )
        if linear_hidden_size is None:
            linear_hidden_size = embedding_size * 4
            self.use_residual = True
        else:
            self.use_residual = False
        self.ffwd = FeedForwardWithCompressionState(
            input_size=embedding_size,
            hidden_size=linear_hidden_size,
            dropout=dropout,
            non_linearity=linear_non_linearity,
            device=device,
        )
        self.ln1 = nn.LayerNorm(embedding_size, device=device)
        self.ln2 = nn.LayerNorm(embedding_size, device=device)

    def forward(self, x, state):
        # x = x.to(self.device)
        # state = state.to(self.device)
        x = x + self.sa(self.ln1(x))
        x, state = self.ffwd(self.ln2(x), state)
        return x, state


class MultiHeadSelfAttentionWithCompression(nn.Module):
    """MultiHeadSelfAttention transformer model (Karpathy nanoGPT derivative)

    Note: the embedding size must be divisible by the number of heads

    :param vocab_size: the size of the vocabulary
    :param embedding_size: the size of the input embedding
    :param sequence_len: the length of the input sequence
    :param dropout: the dropout rate
    :param num_heads: the number of attention heads
    :param num_layers: the number of transformer blocks
    :param causal: whether to use causal masking
    :param linear_non_linearity: the non-linearity to use in between the dual-linear layers,
    :param linear_yoke: Tuple (layer_index, hidden_size, linear_yoke_residual) to yoke the linear layer, or None
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
        linear_non_linearity="relu",
        linear_yoke=None,
        device=None,
    ):
        super().__init__()
        if device is None:
            raise ValueError("Device is None at MultiHeadSelfAttentionWithCompression.")
        self.device = device
        self.sequence_len = sequence_len
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            vocab_size, embedding_size, device=device
        )
        self.position_embedding_table = nn.Embedding(
            sequence_len, embedding_size, device=device
        )
        blks = []
        for i in range(num_layers):
            if linear_yoke is not None and linear_yoke[0] == i:
                linear_hidden_size = linear_yoke[1]
                yoke_residual = linear_yoke[2]
            else:
                linear_hidden_size = None
                yoke_residual = True
            if yoke_residual is True:
                blks.append(
                    BlockWithCompression(
                        embedding_size=embedding_size,
                        sequence_len=sequence_len,
                        dropout=dropout,
                        num_heads=num_heads,
                        causal=causal,
                        linear_non_linearity=linear_non_linearity,
                        linear_hidden_size=linear_hidden_size,
                        device=self.device,
                    )
                )
            else:
                blks.append(
                    BlockWithCompressionNoYokeResidual(
                        embedding_size=embedding_size,
                        sequence_len=sequence_len,
                        dropout=dropout,
                        num_heads=num_heads,
                        causal=causal,
                        linear_non_linearity=linear_non_linearity,
                        linear_hidden_size=linear_hidden_size,
                        device=self.device,
                    )
                )
        self.blocks = blks  # nn.Sequential(*blks)
        self.ln_f = nn.LayerNorm(embedding_size, device=device)  # final layer norm
        self.lm_head = nn.Linear(embedding_size, vocab_size, device=device)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)

        # XXX: move to init, make not trainable:
        # if self.device is None:
        #     pos_emb = self.position_embedding_table(torch.arange(T))
        # else:
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)

        x = tok_emb + pos_emb  # (B,T,C)
        # x = x.to(self.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        # x = self.blocks(x)  # (B,T,C)
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


class MultiHeadSelfAttentionWithCompressionState(nn.Module):
    """MultiHeadSelfAttention transformer model (Karpathy nanoGPT derivative)

    Note: the embedding size must be divisible by the number of heads

    :param vocab_size: the size of the vocabulary
    :param embedding_size: the size of the input embedding
    :param sequence_len: the length of the input sequence
    :param dropout: the dropout rate
    :param num_heads: the number of attention heads
    :param num_layers: the number of transformer blocks
    :param causal: whether to use causal masking
    :param linear_non_linearity: the non-linearity to use in between the dual-linear layers,
    :param linear_yoke: Tuple (layer_index, hidden_size, linear_yoke_residual) to yoke the linear layer, or None
    :param yoke_residual: wether the yoke gets a residual connection, default False
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
        linear_non_linearity="tanh",
        linear_yoke=None,
        device=None,
    ):
        super().__init__()
        if device is None:
            raise ValueError(
                "Device is None at MultiHeadSelfAttentionWithCompressionState"
            )
        self.device = device
        self.sequence_len = sequence_len
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            vocab_size, embedding_size, device=device
        )
        self.position_embedding_table = nn.Embedding(
            sequence_len, embedding_size, device=device
        )
        blks = []
        self.yoke_index = -1
        self.zero_state = torch.zeros([]).to(device)
        for i in range(num_layers):
            if linear_yoke is not None and linear_yoke[0] == i:
                self.yoke_index = i
                linear_hidden_size = linear_yoke[1]
                yoke_residual = linear_yoke[2]
            else:
                linear_hidden_size = None
                yoke_residual = True
            if yoke_residual is True:
                blks.append(
                    BlockWithCompressionState(
                        embedding_size=embedding_size,
                        sequence_len=sequence_len,
                        dropout=dropout,
                        num_heads=num_heads,
                        causal=causal,
                        linear_non_linearity=linear_non_linearity,
                        linear_hidden_size=linear_hidden_size,
                        device=device,
                    )
                )
            else:
                blks.append(
                    BlockWithCompressionStateNoYokeResidual(
                        embedding_size=embedding_size,
                        sequence_len=sequence_len,
                        dropout=dropout,
                        num_heads=num_heads,
                        causal=causal,
                        linear_non_linearity=linear_non_linearity,
                        linear_hidden_size=linear_hidden_size,
                        device=device,
                    )
                )
        self.blocks = blks  # nn.Sequential(*blks)
        self.ln_f = nn.LayerNorm(embedding_size, device=device)  # final layer norm
        self.lm_head = nn.Linear(embedding_size, vocab_size, device=device)

    def forward(self, idx, targets=None, state=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)

        # XXX: move to init, make not trainable:
        # if self.device is None:
        #     pos_emb = self.position_embedding_table(torch.arange(T))
        # else:
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)

        x = tok_emb + pos_emb  # (B,T,C)
        for i, blk in enumerate(self.blocks):
            if i != self.yoke_index:
                # x = x.to(self.device)
                x, _ = blk(x, self.zero_state)
            else:
                # x = x.to(self.device)
                x, state = blk(x, state)
        # x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, state

    def generate(self, idx, max_new_tokens, state, temperature=1.0, top_k=None):
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
            state = state[:, -self.sequence_len :, :]
            # print(idx_cond.shape)
            # get the predictions
            logits, loss, state = self(
                idx_cond, state=state[:, -idx_cond.shape[1] :, :]
            )
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
            state = torch.cat((state, state[:, -1:, :]), dim=1)
            state[:, -1, :] = 0
        return idx, state
