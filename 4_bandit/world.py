import numpy as np

class BanditArm:
    ''' Class that implements just the arm of the bandit
    '''
    def __init__(self, p_success):
        self.p_success = p_success

    def pull(self, rng):
        ''' Simulates pulling of bandit arm
            Returns reward = 1 with probability p_success
            Returns reward = 0 with probability (1 - p_success)
        '''
        if rng.random() < self.p_success:
            return 1
        else:
            return 0


class OneArmedBandit:
    ''' Implements the one-armed bandit that contains an object of BanditArm
    '''
    def __init__(self, p_success=0.3):
        self.rng = np.random.default_rng()
        self.arm = BanditArm(p_success)

    def sample(self):
        return self.arm.pull(self.rng)

class TwoArmedBandit:
    def __init__(self, p_success_1=0.3, p_success_2=0.5):
        self.rng = np.random.default_rng()
        self.arms = [BanditArm(p_success_1), BanditArm(p_success_2)]

    def sample(self, decision):
        return self.arms[decision].pull(self.rng)


class SequentialTwoArmedBandit:
    def __init__(self, p_success_state_A=(0.3, 0.1), p_success_state_B=(0.8, 0.4)):
        self.rng = np.random.default_rng()
        self.states = ['A', 'B']
        self.arms = {'A': [BanditArm(p_success_state_A[0]), BanditArm(p_success_state_A[1])],
                     'B': [BanditArm(p_success_state_B[0]), BanditArm(p_success_state_B[1])]}
        self.transitions = {('A',0): 'A', ('A', 1): 'B', ('B', 0): 'A', ('B', 1): 'B'}
        self.current_state = 'A'

    def sample(self, decision):
        reward = self.arms[self.current_state][decision].pull(self.rng) # Simulation step
        next_state = self.transitions[(self.current_state, decision)] # Example: self.transitions[('A',0)]
        self.current_state = next_state
        return reward, next_state