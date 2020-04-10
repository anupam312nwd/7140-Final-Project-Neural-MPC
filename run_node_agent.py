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
    model.load_state_dict(torch.load(model_path)) 
else:
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

    plt.title("Loss history, smoothed over 100 step window")
    plt.plot(smooth(hist, 100))
    plt.savefig("plots/node_train.png")

    test_loss = 0
    with torch.no_grad():
        test_loss = test(model, env)
    print(f"Test loss: {test_loss}")

# Run mpc with learned model
mpc_config = {
    "action_size": env.action_space.n,
    "state_size": 4,
    "horizon": 10, 
    "iters": 10,
    "num_candidates": 10,
}

states, costs = run_mpc(model.predict_horizon, lambda s: np.linalg.norm(s), mpc_config, env)

plt.title(f"Cost horizon, over {mpc_config['horizon']}")
plt.plot(costs)
plt.savefig("plots/node_mpc_cost.png")
