import gym
from tqdm import tqdm 

env = gym.make("Acrobot-v1")
observation = env.reset()

pbar = tqdm(range(1000))
for _ in pbar:
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  pbar.set_description(f"Reward: {reward}")

  if done:
    observation = env.reset()
env.close()