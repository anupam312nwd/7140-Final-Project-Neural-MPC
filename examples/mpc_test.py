import numpy as np
import gym
import matplotlib.pyplot as plt

from utils.model_predictive_controller import run_mpc
from utils.acrobot_env import AcrobotEnv


env = AcrobotEnv()

def cost_up(states):
    """ Cost function to get acrobot upright"""
    return np.sum(abs(-np.cos(states[0]) - np.cos(states[1]) + states[0]) - 1)

def cost_down(states):
    """ Cost function to get acrobot straight down"""
    return np.linalg.norm(states) 

def predict_horizon(state, action_sequence):
    """ Takes the env model and runs it on several actions"""
    h = action_sequence.shape[0]
    prediction = np.zeros((4, h))

    prediction[:, 0] = env.model(state, action_sequence[0])
    for i, action in enumerate(action_sequence[1:]):
        ns = env.model(state, action)
        prediction[:, i + 1] = env.model(state, action)

        state = ns
    
    return prediction

config = {
    "action_size": env.action_space.n,
    "state_size": 4,
    "horizon": 500,
    "iters": 5,
    "max_iters": 100,
    "num_candidates": 10
}

states, costs = run_mpc(predict_horizon, cost_up, config, env, video=True, seed_state=[1, 0, 0, 0])

plt.plot(costs)
plt.title("Test mpc go to origin")
plt.savefig("plots/mpc_test_origin.png")