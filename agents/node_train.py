from utils.plot_utils import generate_video

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint

mse = nn.MSELoss()

class TransitionModelNet(nn.Module):
    def __init__(self):
        super(TransitionModelNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(5, 10),
            nn.ELU(),
            nn.Linear(10, 20),
            nn.ELU(),
            nn.Linear(20, 5),
        )
    
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        
    def forward(self, t, y):
        return self.net(y)
    
    def predict_horizon(self, state, action_sequence):
        horizon = len(action_sequence)
        state = torch.Tensor(state)
        states = torch.zeros((state.shape[0], horizon))
        for i, a in enumerate(action_sequence):
            s_augmented = torch.cat((state, torch.Tensor([a]).float()))
            ns = odeint(self, s_augmented, torch.Tensor([0, 0.2]))[1, :4]

            states[:, i] = ns
            state = ns
        
        return states.detach().numpy()


def get_batch(data, data_size, batch_size):
    s = torch.from_numpy(np.random.choice(data_size, batch_size, replace=False))
    return data[0][s], data[1][s], data[2][s]

def train(model, data, config={"horizon": 10, "iters": 1000, "batch_size": 32}):
    horizon = config["horizon"]
    batch_size = config["batch_size"]
    iters = config["iters"]
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    states, actions, next_states = data
    n_samples = states.shape[0]
    data_size = int(n_samples/2)
    data_train = states[0:data_size,:], actions[0:data_size,:], next_states[0:data_size,:]
    data_test = states[data_size:,:], actions[data_size:,:], next_states[data_size:,:]

    print("Training... ")
    loss_history = np.zeros((iters, 2))
    pbar = tqdm(range(iters))
    for it in pbar:
        optimizer.zero_grad()
        state_batch, action_batch, next_state_batch = get_batch(data_train, data_size, batch_size)
        augmented_state_batch = torch.cat([state_batch, action_batch], dim=1)

        pred_state = odeint(model, augmented_state_batch, torch.Tensor([0, 0.2]))

        loss = mse(next_state_batch.squeeze(1), pred_state[1, :, :4])
        loss_history[it, 0] = loss.item()

        loss.backward()
        optimizer.step()


        with torch.no_grad():
            state_batch, action_batch, next_state_batch = get_batch(data_test, data_size, batch_size)
            augmented_state_batch = torch.cat([state_batch, action_batch], dim=1)
            pred_state = odeint(model, augmented_state_batch, torch.Tensor([0, 0.2]))

            loss = mse(next_state_batch.squeeze(1), pred_state[1, :, :4])
            loss_history[it, 1] = loss.item()

        if it % 50 == 0:
            pbar.set_description(f"train loss: {loss_history[it, 0]}, test loss: {loss_history[it, 1]}")


    print("Done.")

    torch.save(model.state_dict(), "agents/models/node_model.pth")

        
    return loss_history, model

def test(model_nn, env, video=False):
    horizon = 100
    actions = torch.randint(0, env.action_space.n, (horizon,))

    env.reset()

    state_true = torch.Tensor(env._get_state())
    state_nn = torch.Tensor(env._get_state())

    states_true = torch.zeros((4, horizon))
    states_nn = torch.zeros((4, horizon))

    imgs = []
    if video:
        imgs.append(env.render_state(state_nn.detach().numpy()))

    for i, a in enumerate(actions):
        s_true_augmented = torch.cat((state_true, a.unsqueeze(0).float()))
        s_nn_augmented = torch.cat((state_nn, a.unsqueeze(0).float()))

        ns_true = odeint(env.dynamics, s_true_augmented, torch.Tensor([0, env.dt]))[1, :4]
        ns_nn = odeint(model_nn, s_nn_augmented, torch.Tensor([0, env.dt]))[1, :4]

        if video:
            imgs.append(env.render_state(ns_true.detach().numpy()))

        states_true[:, i] = ns_true
        states_nn[:, i] = ns_nn

        state_true = ns_true
        state_nn = ns_nn

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    fig.suptitle(f"NODE: Predictions over {horizon} steps")

    ax0.plot(states_true[0, :], label="True")
    ax0.plot(states_nn[0, :].detach(), label="Predicted")
    ax0.set_title(r'$\theta_0$')

    ax1.plot(states_true[1, :], label="True")
    ax1.plot(states_nn[1, :].detach(), label="Predicted")
    ax1.legend(loc="lower right")
    ax1.set_title(r'$\theta_1$')

    ax2.plot(states_true[2, :], label="True")
    ax2.plot(states_nn[2, :].detach(), label="Predicted")
    ax2.set_title(r'$\dot{\theta}_0$')

    a = ax3.plot(states_true[0, :], label="True")
    b = ax3.plot(states_nn[0, :].detach(), label="Predicted")
    ax3.set_title(r'$\dot{\theta}_1$')

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    plt.savefig("plots/node_train_true_compare.png")

    if video:
        print("Generating test video...")
        generate_video(imgs, "plots/node_test_video.gif")
        print("Done.")
        env.close()
    
    return (state_true - state_nn).pow(2) / state_true.shape[0]