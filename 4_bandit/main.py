from world import OneArmedBandit, TwoArmedBandit
from agent import Agent
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == "__main__":
    n_samples = 2000

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