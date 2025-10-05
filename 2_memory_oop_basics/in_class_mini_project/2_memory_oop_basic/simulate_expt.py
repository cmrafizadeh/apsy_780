import argparse
from .models import SimpleMemoryModel, InterferenceMemoryModel
from .sim_group import simulate_group
from .plot_data import create_myplot

def run(all_my_arguments):
    someones_memory = SimpleMemoryModel() # Individual 1's memory instance
    someone_elses_memory=InterferenceMemoryModel(seed=32) # Individual 2's memory instance -- this participant's memory is worse. If this penalty wasn't directly specified here, it would've used the default value specified in line 32.

    #print(type(someone_elses_memory.rng))



    for trial in range(all_my_arguments.num_trials):
        p1, acc1, = someones_memory.simulate_trial(all_my_arguments.list_of_letters)
        print(f"Prob & Acc of recall for Individual 1:{p1}, {acc1}")

        p2, acc2 = someone_elses_memory.simulate_trial(all_my_arguments.list_of_letters)
        print(f"Prob & Acc of recall for Individual 2:{p2}, {acc2}")

if __name__=='__main__': # If the name assigned to the current module I'm running is main (needs to be specified from command line), then do the following...
    
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--list_length', type=int, nargs='+', default=['4 6 8'])
    my_parser.add_argument('--num_trials', type=int, default=20)

    #print(type(all_my_arguments.list_of_letters))


    my_parser.add_argument('--num_participants', type=int, default=16)
    all_my_arguments = my_parser.parse_args()
    simulation=simulate_group(num_participants=all_my_arguments.num_participants, list_length=all_my_arguments.list_length, num_trials=all_my_arguments.num_trials, file_name='letter_memory.csv')
    create_myplot('letter_memory.csv')
    

    