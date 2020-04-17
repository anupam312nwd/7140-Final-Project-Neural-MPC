"""GRU implementation for predicting state"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
# %matplotlib inline

parser = argparse.ArgumentParser('vGRU demo')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

data_states = np.load('data/trial1_states.npy')
data_actions = np.load('data/trial1_actions.npy')
data_next_states = np.load('data/trial1_next_states.npy')
input_data = torch.from_numpy(np.concatenate((data_states, data_actions), axis=1))
output_data = torch.from_numpy(data_next_states)

N = input_data.shape[0]
print(N)
data = []
for i in range(N):
    data.append((input_data[i], output_data[i]))
print(len(data))

train_data = data[:int(2/3 * N)]
test_data = data[int(2/3 * N):]
# print(len(train_data), len(test_data))


class vGRU(nn.Module):

    def __init__(self, input_size=7, hidden_size=30, out_size=6):
        super().__init__()
        self.hidden_size = hidden_size
        # Add an vGRU layer
        self.gru = nn.GRU(input_size, hidden_size)
        # Add a fully-connected layer
        self.linear = nn.Linear(hidden_size, out_size)
        # Initialize h0
        # self.hidden = torch.zeros(1, 1, hidden_size)
        self.hidden = (torch.zeros(1, 1, hidden_size))

    def forward(self, seq):
        gru_out, self.hidden = self.gru(seq.reshape(-1, 1, len(seq)), self.hidden)
        pred = self.linear(gru_out.reshape(1, -1))
        return pred[-1]   # we only care about the last prediction


model = vGRU()
print(model)
print('---------------------------------')
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')


count_parameters(model)

epochs = 20
train_loss = []
test_loss = []
N_train, N_test = len(train_data), len(test_data)

for i in range(epochs):

    # tuple-unpack the train_data set
    sum_loss = 0
    for seq, y_train in train_data:

        # reset the parameters and hidden states
        seq = seq.float()
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size))

        y_pred = model(seq)

        loss = criterion(y_pred.float(), y_train.float())
        sum_loss += loss
        loss.backward()
        optimizer.step()

    train_loss.append(sum_loss/N_train)
    # print training result
    print(f'Epoch: {i+1:2} Training Loss: {loss.item():10.8f}')

    with torch.no_grad():
        sum_loss = 0
        for seq, y_test in test_data:

            seq = seq.float()
            model.hidden = (torch.zeros(1, 1, model.hidden_size))
            y_pred = model(seq)
            loss = criterion(y_pred.float(), y_test.float())
            sum_loss += loss

        test_loss.append(sum_loss/N_test)
        # print testing result
        print(f'Epoch: {i+1:2} Testing Loss: {loss.item():10.8f}')

plt.plot(train_loss, label="train_loss")
plt.plot(test_loss, label="test_loss")
plt.legend()
plt.savefig("Train_test_loss.png")
plt.show()

torch.save(model.state_dict(), 'node_agent/sample_torchgru.pt')

