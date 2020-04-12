from agents.node_train import TransitionModelNet, train, test
from utils.acrobot_env import AcrobotEnv
from utils.gym_utils import generate_torch_training_data
from utils.plot_utils import smooth
from utils.model_predictive_controller import run_mpc

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model_path = "agents/models/node_model.pth"

env = AcrobotEnv()
model = TransitionModelNet()

# Check if we already have a trained model
if os.path.isfile(model_path):
    print("Loading model file")
    model.load_state_dict(torch.load(model_path)) 
else:
    print("No model file, training new model")
    model_config = {
        "n_samples": 50000,
        "horizon": 1,
        "iters": 10000,
        "batch_size": 64,
    }

    data = generate_torch_training_data(env, model_config["n_samples"], filename="data/")
    hist, model = train(
        model.to(device), 
        (data[0].to(device), data[1].to(device), data[2].to(device)), 
        model_config
    )

    plt.figure()
    plt.title("Loss history, smoothed over 100 step window")
    plt.plot(smooth(hist[:, 0], 200))
    plt.plot(smooth(hist[:, 1], 200))
    plt.savefig("plots/node_test_train.png")
    np.save("data/node_train_loss.npy", hist[:, 0])
    np.save("data/node_test_loss.npy", hist[:, 1])

test_loss = 0
with torch.no_grad():
    test_loss = test(model, env)

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
plt.savefig("plots/node_mpc_cost.png")
