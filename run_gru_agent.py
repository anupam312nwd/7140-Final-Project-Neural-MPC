#!/usr/bin/env python3

from agents.gru_train import vGRU, train, test
from utils.acrobot_env import AcrobotEnv
from utils.gym_utils import generate_torch_training_data
from utils.plot_utils import smooth
from utils.model_predictive_controller import run_mpc

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "agents/models/gru_model.pth"
device = 'cpu'

batch = 64
env = AcrobotEnv()
model = vGRU(batch_size=batch)

# Check if we already have a trained model
if os.path.isfile(model_path):
    print("Loading model file")
    model.load_state_dict(torch.load(model_path))
else:
    print("No model file, training new model")
    config = {
        "n_samples": 50000,         # 10000
        "horizon": 1,
        "iters": 10000,              # 1000
        "batch_size": batch,
    }

    data = generate_torch_training_data(env, config["n_samples"], filename="data/")

    hist_train, hist_test, model = train(
        model.to(device),
        (data[0].to(device), data[1].to(device), data[2].to(device)),
        config
    )

    plt.plot(smooth(hist_train, 200), label='train')
    plt.plot(smooth(hist_test, 200), label='test')
    plt.legend()
    plt.savefig("plots/gru_train_test.png")
    np.save("data/gru_train_loss.npy", hist_train)
    np.save("data/gru_test_loss.npy", hist_test)

test_loss = 0
with torch.no_grad():
    test_loss = test(model, env, video=False)

print(f"Test loss: {test_loss}")

# Run mpc with learned model
mpc_config = {
    "action_size": env.action_space.n,
    "state_size": 4,
    "horizon": 50,
    "iters": 5,
    "num_candidates": 4,
    "max_iters": 100,
}

states, costs = run_mpc(model.predict_horizon, lambda s: np.linalg.norm(-np.cos(s[0]) - np.cos(s[1] + s[0]) - 1), mpc_config, env)

plt.figure()
plt.title(f"Cost, mpc horizon: {mpc_config['horizon']}")
plt.plot(smooth(costs, 10))
plt.xlabel("Environment Step")
plt.ylabel("Cost (Distance from swing up)")
plt.savefig("plots/gru_mpc_cost.png")
