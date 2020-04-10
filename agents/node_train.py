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

def get_batch(data, data_size, batch_size):
    s = torch.from_numpy(np.random.choice(data_size, batch_size, replace=False))
    return data[0][s], data[1][s], data[2][s]

def train(model, data, config={"horizon": 10, "iters": 1000, "batch_size": 32}):
    horizon = config["horizon"]
    batch_size = config["batch_size"]
    iters = config["iters"]
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    states, actions, next_states = data
    data_size = states.shape[0]
    print(f"Data dimensions: \n{states.shape}, \n{actions.shape}, \n{next_states.shape}")


    print("Training... ")
    loss_history = np.zeros(iters)
    pbar = tqdm(range(iters))
    for it in pbar:
        optimizer.zero_grad()
        state_batch, action_batch, next_state_batch = get_batch(data, data_size, batch_size)
        augmented_state_batch = torch.cat([state_batch, action_batch], dim=1)

        pred_state = odeint(model, augmented_state_batch, torch.Tensor([0, 0.2]))

        loss = mse(next_state_batch.squeeze(1), pred_state[1, :, :4])
        loss_history[it] = loss.item()

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


    states_true = torch.zeros((4, horizon))
    states_nn = torch.zeros((4, horizon))

    for i, a in enumerate(actions):
        s_true_augmented = torch.cat((state_true, a.unsqueeze(0).float()))
        s_nn_augmented = torch.cat((state_nn, a.unsqueeze(0).float()))

        ns_true = odeint(env.dynamics, s_true_augmented, torch.Tensor([0, env.dt]))[1, :4]
        ns_nn = odeint(model_nn, s_nn_augmented, torch.Tensor([0, env.dt]))[1, :4]


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

