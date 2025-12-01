from world import OneArmedBandit, TwoArmedBandit, SequentialTwoArmedBandit
from agent import Agent, DoubleOZero
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

if __name__ == "__main__":
    n_samples = 3000

    choice = 'seq_bandit'

    if choice == 'two_armed':
        # Create bandit
        # bandit = OneArmedBandit()
        bandit = TwoArmedBandit()
        agent = Agent(alpha=0.01)

        # Simulate bandit and record rewards
        rewards = []
        q_values_slot1 = []
        q_values_slot2 = []
        for ii in range(n_samples):
            decision = agent.make_decision()
            reward_ii = bandit.sample(decision)
            agent.update(reward_ii, decision)

            rewards.append(reward_ii)
            q_values_slot1.append(agent.q_values[0])
            q_values_slot2.append(agent.q_values[1])
            # rewards.append(bandit.sample())

        # rewards = [bandit.sample() for _ in range(n_samples)] 
        # list_of_events = [ii+1 for ii in range(n_samples)]

        print(f"Rewards = {rewards}")
        print(f"Q-values = {q_values_slot1}")
        print(f"Q-values = {q_values_slot2}")
        # print(f"List of events = {list_of_events}")
        print(f"Average reward = {np.mean(rewards)}")

        # sns.lineplot(range(n_samples), q_values)
        sns.lineplot(q_values_slot1)
        sns.lineplot(q_values_slot2)
        plt.show()

    elif choice == 'seq_bandit':
        # Create bandit
        seq_bandit = SequentialTwoArmedBandit()
        new_agent = DoubleOZero()

        # Simulate bandit and record rewards
        rewards = []
        q_values = pd.DataFrame(columns=['A_0', 'A_1', 'B_0', 'B_1'])
        for ii in range(n_samples):
            state_t = seq_bandit.current_state
            decision = new_agent.make_decision(state_t)
            reward_ii, next_state = seq_bandit.sample(decision)
            new_agent.update(reward=reward_ii, decision=decision, state_t=state_t, state_t_plus_1=next_state)

            # Store and display
            rewards.append(reward_ii)
            q_values.loc[ii] = [new_agent.q_values['A'][0], new_agent.q_values['A'][1],
                                new_agent.q_values['B'][0], new_agent.q_values['B'][1]]
            # rewards.append(bandit.sample())

        # sns.lineplot(range(n_samples), q_values)
        sns.lineplot(data=q_values, x=q_values.index, y='A_0', label='A_0')
        sns.lineplot(data=q_values, x=q_values.index, y='A_1', label='A_1')
        sns.lineplot(data=q_values, x=q_values.index, y='B_0', label='B_0')
        sns.lineplot(data=q_values, x=q_values.index, y='B_1', label='B_1')
        plt.legend()
        plt.show()
