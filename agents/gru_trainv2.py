"""GRU training model update"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import argparse
# %matplotlib inline

parser = argparse.ArgumentParser('vGRU demo')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


class vGRU(nn.Module):

    def __init__(self, input_size=5, hidden_size=30, out_size=4, batch_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        # Add an vGRU layer
        self.gru = nn.GRU(input_size, hidden_size)
        # Add a fully-connected layer
        self.linear = nn.Linear(hidden_size, out_size)
        # Initialize h0
        self.hidden = torch.zeros(1, 1, hidden_size)

    def forward(self, seq, hidden):
        # print('sequence info:', type(seq), len(seq))
        gru_out, hidden = self.gru(seq.reshape(len(seq), self.batch_size,
                                                    self.input_size), hidden)
        pred = self.linear(gru_out)  # .reshape(1, -1)
        return pred[-1], hidden   # we only care about the last prediction


model = vGRU()
print(model)
print('---------------------------------')
criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


def get_batch(data, data_size, batch_size):
    s = torch.from_numpy(np.random.choice(data_size, batch_size, replace=False))
    # print(data[0][s].shape, data[1][s].shape, data[2][s].shape)
    # (64,4), (64,1), (64,5)
    return data[0][s], data[1][s], data[2][s]


def train(model, data, config={"horizon": 1, "iters": 1000, "batch_size": 32}):
    horizon = config["horizon"]
    batch_size = config["batch_size"]
    iters = config["iters"]
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    states, actions, next_states = data
    data_size = states.shape[0]
    print(f"Data dimensions: \n{states.shape}, \n{actions.shape}, \n{next_states.shape}")

    print("Training... ")
    loss_history = np.zeros(iters)
    pbar = tqdm(range(iters))
    # model.hidden = (torch.zeros(1, model.batch_size, model.hidden_size))
    hidden = model.hidden
    for it in pbar:
        state_batch, action_batch, next_state_batch = get_batch(data, data_size, batch_size)
        augmented_state_batch = torch.cat([state_batch, action_batch], dim=1)
        augmented_state_batch = augmented_state_batch.view(1, augmented_state_batch.shape[0], augmented_state_batch.shape[1])

        optimizer.zero_grad()
        if it==1: print('augmented_state_batch', augmented_state_batch.shape)
        sum_loss = 0
        for seq in augmented_state_batch:
            pred_state, hidden = model(seq.float(), hidden)
            hidden = hidden.detach()
            loss = criterion(pred_state.float(), next_state_batch.float())
            sum_loss += loss
            loss.backward()
            optimizer.step()
        # loss_history[it] = loss.item()
        loss_history[it] = sum_loss.item()
        # loss = mse(next_state_batch.squeeze(1), pred_state[1, :, :4])

        pbar.set_description(f"Current loss: {loss.item()}")
    print("Done!")

    torch.save(model.state_dict(), "agents/models/node_model.pth")
    return loss_history, model


def test(model_nn, env):
    horizon = 100
    actions = torch.randint(0, env.action_space.n, (horizon,))

    env.reset()

    state_true = torch.Tensor(env._get_state())
    state_nn = torch.Tensor(env._get_state())

    # model.hidden = (torch.zeros(1, model.batch_size, model.hidden_size))

    states_true = torch.zeros((4, horizon))
    states_nn = torch.zeros((4, horizon))

    hidden = model.hidden
    for i, a in enumerate(actions):
        # print(state_nn, state_true, a)
        s_true_augmented = torch.cat((state_true.float(), a.unsqueeze(0).float()))
        s_nn_augmented = torch.cat((state_nn.float(), a.unsqueeze(0).float()))

        # ns_true = odeint(env.dynamics, s_true_augmented, torch.Tensor([0, env.dt]))[1, :4]
        # ns_nn = odeint(model_nn, s_nn_augmented, torch.Tensor([0, env.dt]))[1, :4]
        ns_true = env._get_state()
        ns_true = torch.from_numpy(ns_true)
        # print('s_nn_augmented', s_nn_augmented.shape)  # torch.Size([5])
        ns_nn, hidden = model(s_nn_augmented.view(1, 1, 5).float(), hidden)

        states_true[:, i] = ns_true
        states_nn[:, i] = ns_nn

        state_true = ns_true
        state_nn = ns_nn[0]
        # print('state_nn', state_nn)

    plt.figure()
    plt.title("State 0 over 10 steps, true vs nn model")
    plt.plot(states_true[0, :], label="theta_true")
    plt.plot(states_nn[0, :].detach(), label="theta_nn")
    plt.legend()
    plt.savefig("plots/gru_train_true_compare.png")

    return (state_true - state_nn).pow(2) / state_true.shape[0]
