import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, D=32, F = 32, H = 4):
        super().__init__()
        # Reminder: Q will have shape (B, N, D), K will have shape (B, M, D), V will have shape (B, M, F)
        self.W_Q = nn.parameter.Parameter(torch.empty(H, D, D))
        self.W_K = nn.parameter.Parameter(torch.empty(H, D, D))
        self.W_V = nn.parameter.Parameter(torch.empty(H, F, F))
        nn.init.xavier_normal_(self.W_Q)
        nn.init.xavier_normal_(self.W_K)
        nn.init.xavier_normal_(self.W_V)

        # O projects back from a shape of (B, H, N, F) to (B, N, F*H)
        self.O = nn.parameter.Parameter(torch.empty(F*H, F))
        nn.init.xavier_normal_(self.O)


    def forward(self, X_Q, X_K, X_V, mask = None):
        # Say we have q queries, k keys
        # In the unbatched case, we have Q = (q, d), K = (k, d), V = (k, f)
        # We want to compute QK^T = (q, k)
        # We can do this with einsum while also keeping track of batches b and heads h
        Q_KT = torch.einsum("bqd, hdd, bkd, hdd -> bhqk", X_Q, self.W_Q, X_K, self.W_K) / torch.sqrt(torch.tensor(X_Q.shape[2]))
        # Now we need to mask the upper triangle of QK^T
        # We can do this by setting the upper triangle to -inf on the last two dimensions of QK^T
        if mask is not None:
            Q_KT.masked_fill_(mask == 0, float("-inf"))
        # Now we can compute the softmax on the k dimension (rows)
        Q_KT = torch.softmax(Q_KT, dim=3)
        # Now we can compute the attention
        # In the unbatched case, QK^T = (q, k), V = (k, f)
        # Q^KT * V = (q, f)
        #V = torch.einsum("bkf, hff -> bhkf", X_V, self.W_V) 
        #O = torch.einsum("bhqk, bhkf -> bhqf", Q_KT, V)       
        O = torch.einsum("bhqk, hff, bkf -> bhqf", Q_KT, self.W_V, X_V)
        # Now we can concatenate the heads
        # O = (b, h, q, f) -> O = (b, q, h*f)
        O = O.transpose(1, 2)
        O = O.reshape(O.shape[0], O.shape[1], O.shape[2] * O.shape[3])
        # Now we can project back to the original dimension
        return torch.einsum("bqx, xf -> bqf", O, self.O)


# Uses self attention to compute the output
class Transformer(nn.Module):
    def __init__(self, tokens, D=32, H = 4, L = 4, positional_encoding = None, mask = None):
        super().__init__()
        self.tokens = tokens
        self.positional_encoding = positional_encoding
        self.mask = mask
        self.embeddings = nn.Embedding(tokens, D)
        self.attentions = nn.modules.ModuleList([AttentionLayer(D, D, H) for _ in range(L)])
        self.layer_norms = nn.modules.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.networks = nn.modules.ModuleList([
            nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D)) 
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
        