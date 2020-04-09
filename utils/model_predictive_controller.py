import numpy as np
import numpy.random as npr

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

        self.action_dist = np.ones(self.action_size) * (1 / self.action_size)
        self.best_action_seq = np.zeros(self.horizon)
    
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
                actions = self.sample_actions(use_action_dist=True)
            
            # actions should already be valid

            predictions = self.predict(state, actions)
            costs = np.array([self.cost(pred) for pred in predictions])

            best_seq = actions[np.argmax(costs)]
            self.update_action_dist(best_seq)

            if np.argmax(costs) > self.cost(self.model(state, self.best_action_seq)):
                self.best_action_seq = best_seq
        
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
        
        if use_action_dist:
            return npr.choice(len(self.action_dist), p=self.action_dist, size=(self.num_candidates, self.horizon))
        else:
            return npr.choice(self.action_size, size=(self.num_candidates, self.horizon))

    def update_action_dist(self, action_seq):
        """ Updates the running distribution of actions to sample"""
        _, counts_elements = np.unique(action_seq, return_counts=True)
        self.action_dist = counts_elements / counts_elements.sum()
