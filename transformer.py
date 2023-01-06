import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, D=32, F = 32, H = 4):
        super().__init__()
        assert(D % H == 0 and F % H == 0)
        self.D_H = D // H
        self.F_H = F // H

        # Reminder: Q will have shape (B, N, D), K will have shape (B, M, D), V will have shape (B, M, F)
        self.W_Q = nn.parameter.Parameter(torch.empty(H, self.D_H, self.D_H))
        self.W_K = nn.parameter.Parameter(torch.empty(H, self.D_H, self.D_H))
        self.W_V = nn.parameter.Parameter(torch.empty(H, self.F_H, self.F_H))
        nn.init.xavier_normal_(self.W_Q)
        nn.init.xavier_normal_(self.W_K)
        nn.init.xavier_normal_(self.W_V)

        # O recombines the heads
        self.O = nn.parameter.Parameter(torch.empty(F, F))
        nn.init.xavier_normal_(self.O)


    def forward(self, X_Q, X_K, X_V, mask = None):
        # X_Q = (b, q, d), similarly other inputs
        # We need to split the last dimension into heads 
        # X_Q = (b, q, d) -> X_Q = (b, q, h, d/h)
        X_Q = X_Q.reshape(X_Q.shape[0], X_Q.shape[1], X_Q.shape[2] // self.D_H, self.D_H)
        X_K = X_K.reshape(X_K.shape[0], X_K.shape[1], X_K.shape[2] // self.D_H, self.D_H)
        X_V = X_V.reshape(X_V.shape[0], X_V.shape[1], X_V.shape[2] // self.F_H, self.F_H)

        # Say we have q queries, k keys
        # In the unbatched case, we have Q = (q, d), K = (k, d), V = (k, f)
        # We want to compute QK^T = (q, k)
        # We can do this with einsum while also keeping track of batches b and heads h
        Q_KT = torch.einsum("bqhd, hdd, bkhd, hdd -> bhqk", X_Q, self.W_Q, X_K, self.W_K) / torch.sqrt(torch.tensor(X_Q.shape[3]))
        # Now we need to mask the upper triangle of QK^T
        # We can do this by setting the upper triangle to -inf on the last two dimensions of QK^T
        if mask is not None:
            Q_KT.masked_fill_(mask == 0, float("-inf"))
        # Now we can compute the softmax on the k dimension (rows)
        Q_KT = torch.softmax(Q_KT, dim=3)
        # Now we can compute the attention
        # In the unbatched case, QK^T = (q, k), V = (k, f)
        # Q^KT * V = (q, f)
        Y = torch.einsum("bhqk, bkhf, hff -> bqhf", Q_KT, X_V, self.W_V)
        # Now we need to recombine the heads
        # Y = (b, q, h, f/h)
        Y = Y.reshape(Y.shape[0], Y.shape[1], Y.shape[2] * Y.shape[3])
        # Y = (b, q, f)
        return torch.einsum("bqf, ff -> bqf", Y, self.O)


# Uses self attention to compute the output
class Transformer(nn.Module):
    def __init__(self, tokens, D=32, H = 4, L = 4, positional_encoding = None, mask = None):
        super().__init__()
        assert(D % H == 0)
        self.tokens = tokens
        self.positional_encoding = positional_encoding
        self.mask = mask
        self.embeddings = nn.Embedding(tokens, D)
        self.attentions = nn.modules.ModuleList([AttentionLayer(D, D, H) for _ in range(L)])
        self.layer_norms = nn.modules.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.networks = nn.modules.ModuleList([
            nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)) 
            for _ in range(L)])
        self.logit_weights = nn.parameter.Parameter(torch.empty(D, tokens))
        nn.init.xavier_normal_(self.logit_weights)

    def forward(self, X):
        # Embed tokens
        X = self.embeddings(X)
        # Add positional encoding, if any
        if self.positional_encoding is not None:
            X = X + self.positional_encoding(X.shape[1])
        # Apply attention layers
        for i in range(len(self.attentions)):
            # Use pre-norm
            # X has shape (B, N, D)
            X_norm = self.layer_norms[i](X)
            Y = self.attentions[i](X_norm, X_norm, X_norm, self.mask)
            # Apply neural network on each row of Y
            Y = self.networks[i](Y)
            X = X + Y
        # Output logits for tokens
        return torch.einsum("bqd, dt -> bqt", X, self.logit_weights)
        