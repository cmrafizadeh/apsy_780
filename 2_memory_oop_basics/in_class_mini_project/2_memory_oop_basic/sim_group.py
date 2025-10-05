import numpy as np
import pandas as pd
from .models import SimpleMemoryModel, InterferenceMemoryModel

def simulate_group(num_participants, list_length, num_trials, file_name):
    data=[]
    for part in range(num_participants):
        if np.random.random()<0.5:
            condition='simple'
            someones_memory=SimpleMemoryModel()
        else:
            condition='interference'
            someones_memory=InterferenceMemoryModel()
        for ll in list_length:


            for t in range(num_trials):

                p, acc=someones_memory.simulate_trial(list_length=ll)
                data.append({'subject':part+1, 'condition':condition, 'trial':t, 'prob':p, 'accuracy':acc, 'list_length':ll})
    
    
        #print(p, acc, condition)
    

    df=pd.DataFrame(data)
    print(df.head(50))
    df.to_csv(file_name)
    return df


