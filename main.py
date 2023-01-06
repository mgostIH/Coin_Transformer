import torch
import torch.nn as nn
import transformer
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 1
learning_rate = 1e-3
batch_size = 320
sequence_length = 8
N = 1_000_000
loss = nn.CrossEntropyLoss()

force_train = True

# Generate dataset of N sequences of heads/tails with uniform probability of length L
def generate_dataset(N, L):
    U = torch.rand((N, L))
    return torch.bernoulli(U).to(torch.long)

if __name__ == '__main__':
    total_data = generate_dataset(N, sequence_length).to(device)
    # We need to mask out the future tokens
    mask = torch.tril(torch.ones(sequence_length-1, sequence_length-1)).to(torch.int64).to(device)
    

    # input has shape (N, L)
    # output has shape (N, L, 2)
    model = transformer.Transformer(2, D=8, H=4, mask=mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    L = []

    # If we have a saved model, load it instead of training
    # If the file doesn't exist, train the model and save it
    if not force_train:
        try:
            model.load_state_dict(torch.load('model.pt'))
            L = torch.load('losses.pt')
        except FileNotFoundError:
            force_train = True
    if force_train:
        for _ in range(epochs):
            for i in range(0, N, batch_size):
                X = total_data[i:i + batch_size]
                Y = X[:, 1:]
                X = X[:, :-1]
                # CrossEntropyLoss expects (N, C, L) for input and (N, L) for output
                # model(X) has shape (N, L, 2)
                # Y has shape (N, L)
                # We need to transpose model(X) to (N, L, 2)
                Y_hat = model(X)
                l = loss(Y_hat.transpose(1, 2), Y)
                l.backward()
                optimizer.step()
                optimizer.zero_grad()
                L.append(l.item())
            print(L[-1])
        # Save the model and losses
        torch.save(model.state_dict(), 'model.pt')
        torch.save(L, 'losses.pt')

    plt.loglog(L)
    plt.show()

    # Observe prediction from given input
    X = torch.tensor([[1, 1, 1, 1, 0, 1, 0, 1]]).to(device)
    # Y_hat has shape (1, 8, 2)
    Y_hat = model(X[:, :-1])
    print(loss(Y_hat.transpose(1, 2), X[:, 1:]))
    print(torch.softmax(Y_hat, dim = -1))