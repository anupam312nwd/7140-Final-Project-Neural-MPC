import numpy as np

def generate_training_data(env, samples, filename=None):
    """Generates training data in the form of numpy array's for state, action, and next state datasets

    Arguments:

        env (gym.Env): gym environment
        samples (int): number of samples 
        filename (string): if a filename is passed, three arrays will be saved to disk, appending the filename with the right identifier. 
    """

    states = np.zeros((samples, *env.observation_space.shape))
    actions = np.zeros((samples, 1))
    next_states = np.zeros((samples, *env.observation_space.shape))

    state = env.reset()
    for s in range(samples):
        action = env.action_space.sample()
        
        # Collect state and action
        states[s] = state
        actions[s] = action

        next_state, _, done, _ = env.step(action)

        # Collect state and action
        next_states[s] = next_state

        state = next_state
        if done:
            env.reset()
    
    if filename:
        np.save(filename + "_states.npy", states)
        np.save(filename + "_actions.npy", actions)
        np.save(filename + "_next_states.npy", next_states)

    return states, actions, next_states



    