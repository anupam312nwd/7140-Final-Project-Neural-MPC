import gym
from tqdm import tqdm 
import matplotlib.pyplot as plt

# from rnn-agent.rnn import MyNetwork
# from node-agent.node import MyNetwork2

env = gym.make("Acrobot-v1")
observation = env.reset()

# obs_shape = env.observation_space.shape
# action_shape = env.action_space.n

# agent = NeuralMPC(config) / NeuralRNN(config) # running the agent
# net = RNN(state_shape) # running the network
for _ in range(5):
  # action = env.action_space.sample() # your agent here (this takes random actions)
  # prediction = net(observation, action) # network takes in the current image, predicts next one 
  # action = agent.act(observation) # your agent here (this takes random actions)

  _, reward, done, info = env.step(action)
  observation = env.render(mode='rgb_array')

  # agent.update(observation) # In train mode
  # net.fit(loss(prediction, observation)) # Update your neural net, with the new observation

  print(f"{observation}, {action}, {reward}, {done}")

  if done:
    observation = env.reset()
plt.imshow(observation)
plt.show()


env.close()
