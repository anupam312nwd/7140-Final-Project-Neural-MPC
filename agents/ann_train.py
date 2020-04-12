#!/usr/bin/env python3

from utils.plot_utils import generate_video

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):

    def __init__(self, input_size=5, out_size=4):
        super().__init__()
        self.input_size = input_size
        self.out_size = out_size

        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 6)
        self.fc3 = nn.Linear(6, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict_horizon(self, state, action_sequence):
        horizon = len(action_sequence)
        state = torch.Tensor(state)
        states = torch.zeros((state.shape[0], horizon))
        for i, a in enumerate(action_sequence):
            # print(state, torch.Tensor([a]))
            s_augmented = torch.cat((state, torch.Tensor([a]).float()))
            ns = model(s_augmented.view(1, 1, 5).float())
            states[:, i] = ns
            state = ns[0][0]

        return states.detach().numpy()

model = ANN()
print(model)

criterion = nn.MSELoss()


def get_batch(data, data_size, batch_size):
    s = torch.from_numpy(np.random.choice(data_size, batch_size, replace=False))
    # print(data[0][s].shape, data[1][s].shape, data[2][s].shape)
    # (64,4), (64,1), (64,5)
    return data[0][s], data[1][s], data[2][s]


def train(model, data, config={"horizon": 1, "iters": 1000, "batch_size": 32}):
    horizon = config["horizon"]
    batch_size = config["batch_size"]
    iters = config["iters"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    states, actions, next_states = data
    n_samples = states.shape[0]
    data_size = int(n_samples/2)
    data_train = states[0:data_size,:], actions[0:data_size,:], next_states[0:data_size,:]
    data_test = states[data_size:,:], actions[data_size:,:], next_states[data_size:,:]

    print('data shape')
    # print(type(data), type(states), type(actions), type(next_states))
    # print(len(data), states.shape, actions.shape, next_states.shape)
    print(f"Data dimensions: \n{states.shape}, \n{actions.shape}, \n{next_states.shape}")

    print("Training... ")
    loss_history_train = np.zeros(iters)
    loss_history_test = np.zeros(iters)
    pbar = tqdm(range(iters))

    for it in pbar:
        """training of architecture and calculation of train loss"""
        state_batch, action_batch, next_state_batch = get_batch(data_train, data_size, batch_size)
        augmented_state_batch = torch.cat([state_batch, action_batch], dim=1)
        augmented_state_batch = augmented_state_batch.view(1, augmented_state_batch.shape[0], augmented_state_batch.shape[1])

        optimizer.zero_grad()
        sum_loss = 0
        for seq in augmented_state_batch:
            seq = seq.view(1, batch_size, 5)
            pred_state = model(seq.float())
            loss = criterion(pred_state.float(), next_state_batch.squeeze(1).float())
            # if it in {0, 1}: print(pred_state.shape, next_state_batch.shape)
            sum_loss += loss
            loss.backward()
            optimizer.step()
        # loss_history[it] = loss.item()
        loss_history_train[it] = sum_loss.item()
        # loss = mse(next_state_batch.squeeze(1), pred_state[1, :, :4])

        pbar.set_description(f"Current loss: {sum_loss.item()}")

        """calculation of test loss"""
        state_batch1, action_batch1, next_state_batch1 = get_batch(data_test, data_size, batch_size)
        augmented_state_batch1 = torch.cat([state_batch1, action_batch1], dim=1)
        augmented_state_batch1 = augmented_state_batch1.view(1, augmented_state_batch1.shape[0], augmented_state_batch1.shape[1])
        with torch.no_grad():
            sum_loss = 0
            for seq in augmented_state_batch1:
                seq = seq.view(1, batch_size, 5)
                pred_state = model(seq.float())
                loss = criterion(pred_state.float(), next_state_batch1.squeeze(1).float())
                sum_loss += loss
            loss_history_test[it] = sum_loss.item()


    print("Done!")

    torch.save(model.state_dict(), "agents/models/ann_model.pth")
    return loss_history_train, loss_history_test, model


def test(model_nn, env, video=False):
    horizon = 100
    actions = torch.randint(0, env.action_space.n, (horizon,))  # [100]

    env.reset()

    state_nn = torch.Tensor(env._get_state())
    states_true = torch.zeros((4, horizon))  # [4, 100]
    states_nn = torch.zeros((4, horizon))

    imgs = []
    if video:
        imgs.append(env.render_state(state_nn.detach().numpy()))

    for i, a in enumerate(actions):
        # Step learned model
        s_nn_augmented = torch.cat((state_nn.float(), a.unsqueeze(0).float()))
        ns_nn = model.forward(s_nn_augmented.view(1, 1, 5).float())

        # Step true model
        env.step(a)
        ns_true = env._get_state()
        ns_true = torch.from_numpy(ns_true)

        # print(ns_nn)
        # print(ns_true)

        if video:
            imgs.append(env.render_state(ns_true.detach().numpy()))

        # Save state to feed into model
        state_nn = ns_nn[0][0]

        states_true[:, i] = ns_true
        states_nn[:, i] = ns_nn


    plt.figure()
    plt.title("State 0 over 10 steps, true vs nn model")
    plt.plot(states_true[0, :], label="theta_true")
    plt.plot(states_nn[0, :].detach(), label="theta_nn")
    plt.legend()
    plt.savefig("plots/ann_train_true_compare.png")

    if video:
        print("Generating test video...")
        generate_video(imgs, "plots/ann_test_video.gif")
        print("Done.")
        env.close()
    return np.linalg.norm(ns_true - ns_nn)
