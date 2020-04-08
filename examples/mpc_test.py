import numpy as np
import gym

from utils.model_predictive_controller import MPC
from utils.acrobot_env import AcrobotEnv


env = AcrobotEnv()

def cost(states):
    """ Cost function to get acrobot upright"""
    return np.sum(((-np.cos(states[0, :]) - np.cos(states[1, :] + states[0, :])) - 1) ** 2)

def predict_horizon(state, action_sequence):
    """ Takes the env model and runs it on several actions"""
    h = action_sequence.shape[0]
    prediction = np.zeros((4, h))

    prediction[:, 0] = env.model(state, action_sequence[0])
    for i, action in enumerate(action_sequence[1:]):
        prediction[:, i] = env.model(state, action)
    
    return prediction

config = {
    "action_size": env.action_space.n,
    "state_size": 4,
    "horizon": 10,
    "iters": 100,
    "num_candidates": 10
}

# Setup the controller
controller = MPC(predict_horizon, cost, config)


# Run until cost change is low or time out
env.reset()
state = env._get_state()
done = False
costs = [np.Infinity]
iters = 0
while True:
    action = controller.act(state)
    env.step(action)
    ns = env._get_state()
    
    costs.append(cost(ns.reshape(4, 1)))

    state = ns
    if abs(costs[-1] - costs[-2]) < 0.1 or iters > 100:
        break
    iters += 1
    

print(costs)
