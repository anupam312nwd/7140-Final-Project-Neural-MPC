import numpy as np
import torch

def generate_training_data(env, samples, horizon=1, filename=None):
    """Generates training data in the form of numpy array's for state, action, and next state datasets

    Arguments:

        env (gym.Env): gym environment
        samples (int): number of samples 
        filename (string): if a filename is passed, three arrays will be saved to disk, appending the filename with the right identifier. 
    """

    states = np.zeros((samples, *env.observation_space.shape))
    actions = np.zeros((samples, horizon))
    next_states = np.zeros((samples, horizon, *env.observation_space.shape))

    for s in range(samples):
        state = env.reset()
        for i in range(horizon):
            action =  env.action_space.sample()
            
            # Collect state and action
            states[s] = state
            actions[s, i] = action

            next_state, _, done, _ = env.step(action)

            # Collect final state
            next_states[s, i] = next_state

            state = next_state
            if done:
                env.reset()
    
    if filename:
        np.save(filename + "_states.npy", states)
        np.save(filename + "_actions.npy", actions)
        np.save(filename + "_next_states.npy", next_states)

    return states, actions, next_states

def generate_torch_training_data(env, samples, horizon=1, filename=None):
    """Generates training data in the form of numpy array's for state, action, and next state datasets

    Arguments:

        env (gym.Env): gym environment
        samples (int): number of samples 
        filename (string): if a filename is passed, three arrays will be saved to disk, appending the filename with the right identifier. 
    """

    states = torch.zeros((samples, 4))
    actions = torch.zeros((samples, horizon))
    next_states = torch.zeros((samples, horizon, 4))

    for s in range(samples):
        env.reset()
        env.set_state(env.sample_state())
        state = env._get_state()
        for i in range(horizon):
            action =  env.action_space.sample()
            
            # Collect state and action
            states[s] = torch.Tensor(state)
            actions[s, i] = torch.Tensor([action])

            next_state, _, done, _ = env.step(action)

            # Collect final state
            next_states[s, i] = torch.Tensor(env._get_state())

            state = next_state
            if done:
                env.reset()
                env.set_state(env.sample_state())
    
    if filename:
        torch.save(states, filename + "states.pt")
        torch.save(actions, filename + "actions.pt")
        torch.save(next_states, filename + "next_states.pt")

    return states, actions, next_states


    