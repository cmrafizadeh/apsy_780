import numpy as np

class Agent:
    def __init__(self, alpha=0.1, epsilon=0.05):
        self.alpha = alpha
        self.q_values = np.zeros(2) 
        self.rng = np.random.default_rng()
        self.epsilon = epsilon

    def update(self, reward, decision):
        ''' Implement the Rescrola-Wagner rule updating the q-value based on reward
        '''
        pred_error = reward - self.q_values[decision]
        learning = self.alpha * pred_error
        self.q_values[decision] += learning

    def make_decision(self):
        best_action = np.argmax(self.q_values)
        if self.rng.random() < self.epsilon:
            # Explore - naive
            # if self.rng.random() > 0.5:
            #     return 0
            # else:
            #     return 1
            # More interesting exploration
            # Don't choose the best. Choose something else
            list_of_all_actions = list(range(len(self.q_values)))
            list_of_all_actions.remove(best_action)
            exploratory_action = np.random.choice(list_of_all_actions)
            return exploratory_action
        else:
            # Exploit
            return best_action


class DoubleOZero:
    def __init__(self, alpha=0.1, epsilon=0.05, states = ('A', 'B'), gamma=0.9):
        self.alpha = alpha
        self.epsilon = epsilon
        self.states = states
        self.gamma = gamma
        self.rng = np.random.default_rng()
        # self.q_values = {'A': np.zeros(2), 'B': np.zeros(2)}
        self.q_values = {state: np.zeros(2) for state in states}

    def make_decision(self, current_state):
        best_action = np.argmax(self.q_values[current_state])
        # Exploration stuff
        if self.rng.random() < self.epsilon:
            list_of_all_actions = list(range(len(self.q_values[current_state])))
            list_of_all_actions.remove(best_action)
            exploratory_action = np.random.choice(list_of_all_actions)
            return exploratory_action
        else:
            # Exploit
            return best_action

    def update(self, reward, decision, state_t, state_t_plus_1):
        td_error = reward + self.gamma * np.max(self.q_values[state_t_plus_1]) - self.q_values[state_t][decision]
        learning = self.alpha * td_error
        self.q_values[state_t][decision] += learning
        return self.q_values[state_t][decision]
