import transformer
import torch

B = 1
N = 8
D = 16

H = 4

embed = torch.nn.Embedding(2, D)
X = torch.tensor([[0, 0, 1, 0, 0, 1, 0, 1]])
X = embed(X)

mask = torch.tril(torch.ones(X.shape[1], X.shape[1])).to(torch.int64)
Z = torch.zeros((B, N, N))
Z.masked_fill_(mask == 0, float('-inf'))
print(torch.softmax(Z, dim=2))


layer = transformer.AttentionLayer(D, D, H)
print(layer(X, X, X, mask = mask))


# # Try attention
# T = torch.randint(0, 2, (B, N))
# print(T)
# model = transformer.Transformer(2, D=D, H=H)
# print(torch.softmax(model(T), dim=2))