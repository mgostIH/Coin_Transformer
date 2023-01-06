import transformer
import torch

B = 1
N = 8
D = 2

H = 3

X = torch.rand((B, N, D))

mask = torch.tril(torch.ones(X.shape[1], X.shape[1])).to(torch.int64)

layer = transformer.AttentionLayer(D, D, H)
print(layer(X, X, X, mask))