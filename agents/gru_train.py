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
        self.hidden = torch.zeros(1, batch_size, hidden_size)

    def forward(self, seq):
        # print('sequence info:', type(seq), len(seq))
        gru_out, self.hidden = self.gru(seq.reshape(len(seq), self.batch_size,
                                                    self.input_size), self.hidden)
        pred = self.linear(gru_out)  # .reshape(1, -1)
        return pred[-1]   # we only care about the last prediction


model = vGRU()
print(model)
print('---------------------------------')
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


def get_batch(data, data_size, batch_size):
    s = torch.from_numpy(np.random.choice(data_size, batch_size, replace=False))
    # print(data[0][s].shape, data[1][s].shape, data[2][s].shape)
    # (64,4), (64,1), (64,5)
    return data[0][s], data[1][s], data[2][s]


def train(model, data, config={"horizon": 1, "iters": 1000, "batch_size": 32}):
    horizon = config["horizon"]
    batch_size = config["batch_size"]
    iters = config["iters"]
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    states, actions, next_states = data
    data_size = states.shape[0]
    print(f"Data dimensions: \n{states.shape}, \n{actions.shape}, \n{next_states.shape}")
    # (100,4), (1,4), (100,1,4)


    print("Training... ")
    loss_history = np.zeros(iters)
    pbar = tqdm(range(iters))
    for it in pbar:
        optimizer.zero_grad()
        state_batch, action_batch, next_state_batch = get_batch(data, data_size, batch_size)
        augmented_state_batch = torch.cat([state_batch, action_batch], dim=1)
        augmented_state_batch = augmented_state_batch.view(1, augmented_state_batch.shape[0], augmented_state_batch.shape[1])
        # pred_state = odeint(model, augmented_state_batch, torch.Tensor([0, 0.2]))
        # augmented_state_batch <-> train_data (in test_torchgru code)
        # next_state_batch <-> test_data (in test_torchgru code)
        # print(type(augmented_state_batch), augmented_state_batch.shape)
        model.hidden = (torch.zeros(1, model.batch_size, model.hidden_size))
        if it==1: print('augmented_state_batch', augmented_state_batch.shape)
        pred_state = model(augmented_state_batch.float())
        # print('pred_state, next_state', pred_state.shape, next_state_batch.shape)
        loss = criterion(pred_state.float(), next_state_batch.float())
        # loss = mse(next_state_batch.squeeze(1), pred_state[1, :, :4])
        # loss_history[it] = loss.item()
        loss.backward()
        optimizer.step()

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

    # model.hidden = (torch.zeros(1, 1, model.hidden_size))

    states_true = torch.zeros((4, horizon))
    states_nn = torch.zeros((4, horizon))

    for i, a in enumerate(actions):
        s_true_augmented = torch.cat((state_true, a.unsqueeze(0).float()))
        s_nn_augmented = torch.cat((state_nn, a.unsqueeze(0).float()))

        # ns_true = odeint(env.dynamics, s_true_augmented, torch.Tensor([0, env.dt]))[1, :4]
        # ns_nn = odeint(model_nn, s_nn_augmented, torch.Tensor([0, env.dt]))[1, :4]
        ns_true, _, _, _ = env.step(a)
        model.hidden = (torch.zeros(1, model.batch_size, model.hidden_size))
        ns_nn = model(s_nn_augmented.float())

        states_true[:, i] = ns_true
        states_nn[:, i] = ns_nn

        state_true = ns_true
        state_nn = ns_nn

    plt.figure()
    plt.title("State 0 over 10 steps, true vs nn model")
    plt.plot(states_true[0, :], label="theta_true")
    plt.plot(states_nn[0, :].detach(), label="theta_nn")
    plt.legend()
    plt.savefig("plots/nn_train_true_compare.png")

    return (state_true - state_nn).pow(2) / state_true.shape[0]
