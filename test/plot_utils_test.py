import numpy as np
import matplotlib.pyplot as plt
import gym

from utils.plot_utils import model_rollout_acc

def T(state_action_pair):
    """ return a random state for acrobot""" 

    return np.random.uniform(
        low=[-1.0, -1.0, -1.0, -1.0, -12.56637, -28.274334],
        high=[1.0, 1.0, 1.0, 1.0, 12.56637, 28.274334]
    )

env = gym.make("Acrobot-v1")

print("Rollout loss for a random guess")
model_rollout_acc(T, env)

plt.savefig("plots/rollout_acc.png")