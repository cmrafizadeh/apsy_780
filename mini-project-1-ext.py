# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define function to make any given trial either congruent or incongruent with a 50% chance of either:
def generate_rts(n_trials):
    condition=['congruent', 'incongruent']
    trial_decider=np.random.choice(
         a=condition, size=n_trials, replace=True, p=[.5, .5])
    
    all_rts=[]

    for trial in trial_decider:
        #rt_congruent=False
        if trial=='congruent':
            trial_rt = np.random.normal(400, 100, 1).astype(int) [0]
        else:
            trial_rt = np.random.normal(550, 100, 1).astype(int) [0]
        
        all_rts.append(trial_rt)

    results={"trial": np.arange(1, n_trials + 1),
         "condition": trial_decider, 
         "rt":all_rts}
    results_df=pd.DataFrame(results)
    return results_df

# New addition (outer loop) that loops through each individual participant
def run_multiple_ps(n_participant, n_trials): # Function takes the following arguments: n_participants and n_trials
    frames=[] # placeholder for later on
    for p_id in range(1, n_participant + 1): # This adds a value of one to each subsequent participant in order to create multiple unique p_ids (each id is one larger than previous)
        df=generate_rts(n_trials)
        df["participant_id"] =p_id
    frames.append(df)
    return pd.concat(frames)

# This function remains unchanged from the original code
def analyze_rts(data):
     mean_congruent=data[data['condition']=='congruent'].rt.mean()
     mean_incongruent=data[data['condition']=='incongruent'].rt.mean()

     print(f"mean of congruent trials: {mean_congruent}")
     print(f"Mean of incongruent trials: {mean_incongruent}")

#results=generate_rts(100) Commented out this line to specify parameters for multiple participants
# Aggregated group plot specifications
df=run_multiple_ps(n_participant=30, n_trials=80) # This specifies that participant 30 had 80 trials

df.to_csv(
    '/Users/carolinerafizadeh/Desktop/AI Modeling/stroop_data.csv')

analyze_rts(df)

# Group-level aggregate plot
group_plot=df.groupby(["participant_id", "condition"])["rt"].mean().reset_index()
sns.barplot(data=group_plot, x="condition", y="rt", errorbar="sd")  # use ci="sd" for standard deviation
plt.title("Group-level Stroop Effect (mean RT ± SD)")
plt.show()

# Within-participant plot
sns.pointplot(data=group_plot, x="condition", y="rt", hue="participant_id")
plt.title("Within-Participant Stroop Effect (mean RT ± SD)")
plt.show()
