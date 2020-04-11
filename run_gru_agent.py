#!/usr/bin/env python3

from utils.acrobot_env import AcrobotEnv
from utils.gym_utils import generate_torch_training_data
from utils.plot_utils import smooth
# from agents.node_train import TransitionModelNet, train, test
from agents.gru_trainv2 import vGRU, train, test

import matplotlib.pyplot as plt

import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

env = AcrobotEnv()

batch = 64
config = {
    "n_samples": 2000,         # 10000
    "horizon": 1,
    "iters": 10000,              # 1000
    "batch_size": batch,
}

data = generate_torch_training_data(env, config["n_samples"], filename="data/")
model = vGRU(batch_size=batch)

hist_train, hist_test, model = train(
    model.to(device),
    (data[0].to(device), data[1].to(device), data[2].to(device)),
    config
)

plt.plot(smooth(hist_train, 100), label='train')
plt.plot(smooth(hist_test, 100), label='test')
plt.legend()
plt.savefig("plots/gru_train_test.png")

test_loss = 0
with torch.no_grad():
    test_loss = test(model, env)

print(f"Test loss: {test_loss}")
