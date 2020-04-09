import gym
import numpy

from utils.gym_utils import generate_training_data

env = gym.make("Acrobot-v1")
filename = "data/trial1"

states, actions, next_states = generate_training_data(env, 5, filename=filename)

print("States: \n" , states)
print("Actions: \n", actions)
print("Next states: \n", next_states)