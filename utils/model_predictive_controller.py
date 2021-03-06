from utils.plot_utils import generate_video

import numpy as np
import numpy.random as npr
from tqdm import tqdm

class MPC:
    """MPC Class that takes a transition model + cost function to find the best action in  a given state"""
    def __init__(self, transition_model, cost_func, config=None):
        self.model = transition_model
        self.cost = cost_func

        if not config:
            config = {
                "action_size": 2,
                "state_size": 1,
                "horizon": 10,
                "iters": 100,
                "num_candidates": 10
            }
        
        self.action_size = config["action_size"]
        self.state_size = config["state_size"]
        self.horizon = config["horizon"]
        self.iters = config["iters"]
        self.num_candidates = config["num_candidates"]
        self.best_k = self.num_candidates // 2

        self.action_dist = np.ones((self.action_size, self.horizon)) * (1 / self.action_size)
        self.best_action_seq = np.zeros(self.horizon)
        self.best_action_seq_cost = np.Infinity
    
    def act(self, state):
        """ Returns best action under given state

        Args:
            state: array of values representing state of env
        
        Return:
            best_action_seq[0]: first action of best sequence found
        """
        for i in range(self.iters):

            if i == 0:
                actions = self.sample_actions()
            else:
                actions = self.sample_actions(use_action_dist=False)
            
            # actions should already be valid

            predictions = self.predict(state, actions)
            costs = np.array([self.cost(pred) for pred in predictions])

            sorted_costs_indx = np.argsort(costs) # low to high
            best_seq_indx = sorted_costs_indx[0]

            update_seq = actions[sorted_costs_indx[:self.best_k]]
            self.update_action_dist(update_seq)

            best_seq = actions[best_seq_indx]
            if costs[best_seq_indx] < self.best_action_seq_cost:
                self.best_action_seq = best_seq
                self.best_action_seq_cost = costs[best_seq_indx]
        
        return self.best_action_seq[0]

    def predict(self, state, action_sequences):
        """Predicts trajectories for several candidate action sequences 
        using the given transition model

        Args:
            state: array of values representing state of env
            action_sequences: (num_candidates, horizon) array of actions
        
        Return:
            prediction: array of trajectories for each action sequence
        """

        prediction = []
        for candidate in action_sequences:
            prediction.append(self.model(state, candidate))

        return prediction
    
    def sample_actions(self, use_action_dist=False):
        """ Sample candidate action trajectories

        Args:
            use_action_dist (boolean): random actions or use our action distribution we've computed
        
        Return:
            action_sequence: (num_candidates, horizon) shaped array with multiple sequences of actions
        """
        actions = np.zeros((self.num_candidates, self.horizon)) 
        for i in range(self.horizon):
            if use_action_dist:
                actions[:, i] = npr.choice(self.action_size, p=self.action_dist[:, i], size=self.num_candidates)
            else:
                actions[:, i] = npr.choice(self.action_size, size=self.num_candidates)
        
        return actions

    def update_action_dist(self, action_seq):
        """ Updates the running distribution of actions to sample"""

        for h in range(self.horizon):
            unique, counts_elements = np.unique(action_seq[:, h], return_counts=True)

            new_action_dist = counts_elements
            if len(unique) < self.action_size:
                new_action_dist = np.zeros(self.action_size)
                for i, u in enumerate(unique):
                    new_action_dist[int(u)] += counts_elements[i]
            self.action_dist[:, h] = new_action_dist / counts_elements.sum()

def run_mpc(transition_model, cost_func, config, env, seed_state=None, video=False):
    max_iters = config["max_iters"]
    controller = MPC(transition_model, cost_func, config)

    states, costs = [], []

    env.reset()
    if seed_state:
        env.set_state(seed_state)
    else:
        env.set_state(env.sample_state())

    state = env._get_state()

    states.append(state)
    costs.append(cost_func(state))
   
    imgs = []
    if video:
        imgs.append(env.render_state(state))

    iters = 0
    pbar = tqdm(total=max_iters)
    while True:
        action = controller.act(state)
        # action = env.action_space.sample()

        env.step(action)
        ns = env._get_state()

        states.append(ns)
        costs.append(cost_func(ns))

        state = ns

        # if costs[-1] < 0.01 or iters > max_iters:
        if iters > max_iters:
            break
        
        iters += 1
        pbar.update(1)
        pbar.set_description(f"Current cost: {costs[-1]}")

        if video:
            imgs.append(env.render_state(ns))
    
    
    if video:
        print("Generating test video...")
        generate_video(imgs, "plots/mpc_test_video.gif")
        print("Done.")
        env.close()
    
    return states, costs