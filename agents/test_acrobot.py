from utils.gym_utils import generate_training_data

import os
import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import gym
import matplotlib.pyplot as plt

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

# device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# def get_batch():
#     s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
#     batch_y0 = true_y[s]  # (M, D)
#     batch_t = t[:args.batch_time]  # (T)
#     batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    # return batch_y0, batch_t, batch_y

mse = torch.nn.MSELoss()

def controller(actions):
    def discrete_controller(t):
        if round(t.item()) <= len(actions) - 1:
            return torch.unsqueeze(actions[round(t.item())], dim=0)
        
        return torch.Tensor([0.0])
    
    return discrete_controller

class TransitionModel(nn.Module):
    def __init__(self, controller=None):
        super(TransitionModel, self).__init__()

        self.controller = controller

        self.net = nn.Sequential(
            nn.Linear(7, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 6),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        
    def set_controller(self, controller):
        self.controller = controller
    
    def forward(self, t, y):
        return self.net(torch.cat([y, self.controller(t - 1)]))

def trainNODE(config, data):
    horizon = config["horizon"]
    n_samples = config["n_samples"]
    iters = config["iters"]

    states, actions, next_states = data

    print("Generated Data")
    states = torch.Tensor(states)
    actions = torch.Tensor(actions)
    next_states = torch.Tensor(next_states)

    model = TransitionModel()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    print("Training.")
    pbar = tqdm(range(iters))
    loss_iters = np.zeros(iters)
    for itr in pbar:
        optimizer.zero_grad()
        smpl = np.random.randint(n_samples)
        # batch_y0, batch_t, batch_y = get_batch()

        model.set_controller(controller(actions[smpl]))
        pred_y = odeint(model, states[smpl], torch.Tensor(range(horizon)))

        loss = mse(next_states[smpl], pred_y)
        loss.backward()
        loss_iters[itr] = loss.item()

        optimizer.step()

        pbar.set_description(f"Current loss: {loss.item()}")

        # if itr % args.test_freq == 0:
        #     with torch.no_grad():
        #         pred_y = odeint(model, true_y0, t)
        #         loss = torch.mean(torch.abs(pred_y - true_y))
        #         print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
        #         ii += 1

    with torch.no_grad():
        smpl = np.random.randint(n_samples)
        model.set_controller(controller(actions[smpl]))
        pred_y = odeint(model, states[0], torch.Tensor(range(len(actions[0]))), method='dopri5')
    

    print("Done. Plotting.")
    print(states[0])
    print(next_states[0, :, 0])
    plt.figure()
    plt.plot(smooth(loss_iters, 50))
    plt.savefig(f"plots/NODE_loss_horizon_{horizon}.png")
    plt.show()

    plt.figure()
    plt.plot(next_states[0, :, 0], 'b', marker="o",label="true")
    plt.plot(pred_y[:, 0], 'bx',  label="predicted")
    plt.plot(actions[0], 'y+')

    plt.plot(next_states[0, :, 1], 'g', marker="o",label="true")
    plt.plot(pred_y[:, 1], 'gx',  label="predicted")

    plt.plot(next_states[0, :, 2], 'k', marker="o",label="true")
    plt.plot(pred_y[:, 2], 'kx',  label="predicted")

    plt.plot(next_states[0, :, 3], 'r', marker="o",label="true")
    plt.plot(pred_y[:, 3], 'rx',  label="predicted")
    # plt.legend()
    plt.title(f"Neural ODE Agent H = {horizon}")

    plt.savefig(f"plots/NODE_performance_horizon_{horizon}.png")
    plt.show()


    return model

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

if __name__ == "__main__":
    N_SAMPLES = 5000
    HORIZON = 5
    env = gym.make("Acrobot-v1")
    data = generate_training_data(env, N_SAMPLES, horizon=HORIZON)

    print(data[1])

    config = {
        "n_samples": N_SAMPLES,
        "horizon": HORIZON,
        "iters": 500
    }
    model = trainNODE(config, data)
