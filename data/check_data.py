import numpy as np

arr_act = np.load("trial1_actions.npy")
arr_states = np.load("trial1_states.npy")
arr_next_states = np.load("trial1_next_states.npy")
print(arr_act.shape, arr_states.shape, arr_next_states.shape)

train_data = np.concatenate((arr_states, arr_act), axis = 1)
print(train_data.shape)
