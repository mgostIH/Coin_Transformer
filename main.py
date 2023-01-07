import torch
import torch.nn as nn
import transformer
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 100
learning_rate = 1e-3
batch_size = 1_024
sequence_length = 32
N = 1_000_000
D = 32
H = 4
loss = nn.CrossEntropyLoss()


force_train = False

# Generate dataset of N sequences of heads/tails with uniform probability of length L
# Every sequence must start with a start token 2
def generate_dataset(N, L):
    U = torch.rand((N,1))
    X = torch.bernoulli(U.expand((N,L-1))).to(torch.long)
    return torch.cat((torch.ones((N,1)).to(torch.long) * 2, X), dim=1)

if __name__ == '__main__':
    total_data = generate_dataset(N, sequence_length).to(device)
    # We need to mask out the future tokens
    mask = torch.tril(torch.ones(sequence_length-1, sequence_length-1)).to(torch.int64).to(device)


    # input has shape (N, L)
    # output has shape (N, L, 3)
    model = transformer.Transformer(3, D=D, H=H, mask=mask, positional_encoding=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Observe prediction from given input
    X = torch.ones((1, sequence_length-1))
    X = torch.cat((torch.ones((1,1)) * 2, X), dim=1).to(device).to(torch.long)
    # Y_hat has shape (1, 8, 3)
    Y_hat = model(X[:, :-1])
    print(loss(Y_hat.transpose(1, 2), X[:, 1:]))
    print(Y_hat)
    print(torch.softmax(Y_hat, dim = -1))

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
        # In case CTRL+C is pressed stop training and save the model
        try:
            for _ in range(epochs):
                for i in range(0, N, batch_size):
                    X = total_data[i:i + batch_size]
                    Y = X[:, 1:]
                    X = X[:, :-1]
                    # CrossEntropyLoss expects (N, C, L) for input and (N, L) for output
                    # model(X) has shape (N, L, 3)
                    # Y has shape (N, L)
                    # We need to transpose model(X) to (N, L, 3)
                    Y_hat = model(X)
                    l = loss(Y_hat.transpose(1, 2), Y)
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    L.append(l.item())
                print(L[-1])
        except KeyboardInterrupt:
            pass
        # Save the model and losses
        torch.save(model.state_dict(), 'model.pt')
        torch.save(L, 'losses.pt')

    plt.loglog(L)
    plt.show()

    # Observe prediction from given input
    X = torch.ones((1, sequence_length-1))
    X = torch.cat((torch.ones((1,1)) * 2, X), dim=1).to(device).to(torch.long)
    # Y_hat has shape (1, 8, 3)
    Y_hat = model(X[:, :-1])
    print(loss(Y_hat.transpose(1, 2), X[:, 1:]))
    print(Y_hat)
    print(torch.softmax(Y_hat, dim = -1))