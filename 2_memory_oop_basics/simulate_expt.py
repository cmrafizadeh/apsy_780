import argparse
import string
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from .models import SimpleMemoryModel, InterferenceMemoryModel

def simulate_group(num_participants, num_trials, list_lengths):
    results = []
    for participant in range(num_participants):
        if np.random.rand() < 0.5:
            condition = 'simple'
            someones_memory = SimpleMemoryModel()
        else:
            condition = 'interference'
            someones_memory = InterferenceMemoryModel(seed=32)
        
        # Loop over list lengths
        for list_length in list_lengths:
            # Form a random list of alphabet letters of the given length
            list_of_letters = np.random.choice(list(string.ascii_lowercase), size=list_length, replace=False)
            list_of_letters = list_of_letters.tolist()

            for trial in range(num_trials):
                p1, acc1 = someones_memory.simulate_trial(list_of_letters)
                results.append({'participant': participant, 'trial': trial, 'p': p1, 'acc': acc1, 'condition': condition, 'list_length': list_length})
    return results

def plot_memory_model(results):
    # Plot memory change with list length for both conditions
    df = pd.DataFrame(results)
    print(df.head())
    # Store the dataframe in a csv file
    df.to_csv('memory_model_results.csv', index=False)

    # Read data from csv file
    df = pd.read_csv('memory_model_results.csv')
    print(df.head())

    sns.lineplot(df, x='list_length', y='p', hue='condition')
    plt.show()

def run(all_my_arguments):
    # someones_memory = SimpleMemoryModel() # Individual 1's memory instance
    # someone_elses_memory = InterferenceMemoryModel(seed=32) # Individual 2's memory instance

    # print(type(someone_elses_memory.rng))

    # for trial in range(all_my_arguments.num_trials):
    #     p1, acc1 = someones_memory.simulate_trial(all_my_arguments.list_of_letters)
    #     print(f"Prob & Acc of recall for Individual 1: {p1}, {acc1}")

    #     p2, acc2 = someone_elses_memory.simulate_trial(all_my_arguments.list_of_letters)
    #     print(f"Prob & Acc of recall for Individual 2: {p2}, {acc2}")
    results = simulate_group(all_my_arguments.num_participants, all_my_arguments.num_trials, all_my_arguments.list_lengths)
    plot_memory_model(results)

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--list_of_letters', type=str, default='57312')
    my_parser.add_argument('--list_lengths', type=int, nargs='+', default=[4, 5, 6, 7], help='List of integers representing list lengths')
    # my_parser.add_argument('--list_lengths', type=int, nargs='+', default=[4, 5, 6, 7], help='List of integers representing list lengths')
    my_parser.add_argument('--num_trials', type=int, default=20)
    my_parser.add_argument('--num_participants', type=int, default=10)
    all_my_arguments = my_parser.parse_args()

    # Print all the arguments
    print(all_my_arguments)

    run(all_my_arguments=all_my_arguments)
