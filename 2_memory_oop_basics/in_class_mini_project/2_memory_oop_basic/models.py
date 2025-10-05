import numpy as np
from .helpers import clip01 # If you define your own package, you can import them just as you would from numpy (.helpers specifies its a relative path).
# from .helpers import * # Calls all functions in the helpers module

class SimpleMemoryModel:
    def __init__(self, encoding_error=0.1, retrieval_decay=0.05, noise=0.05, seed=None): # Class constructor
        self.retrieval_decay=retrieval_decay # Instance attributes: decay depends on the length of things stored in memory (list_length)
        self.encoding_error=encoding_error
        self.rng=np.random.default_rng(seed)
        self.noise=noise # Memory also has a "noise" attribute that determines the standard deviation in the probability of recall from trial to trial.

    def simulate_trial(self, list_length): 
       prob_correct_recall = (1 - self.encoding_error) - (self.retrieval_decay * list_length) # probability of correct recall if there wasn't any decay
       prob_correct_recall = prob_correct_recall + self.rng.normal(0.0, self.noise) # The probability of correct recall will vary from trial to trial

       # Clip the probabilities between 0 & 1
       prob_correct_recall = clip01(prob_correct_recall)
       
       if prob_correct_recall > self.rng.random():
           accurate_recall=True
       else:
           accurate_recall=False 
           
       return prob_correct_recall, accurate_recall


# Interference Model (Child class)
class InterferenceMemoryModel(SimpleMemoryModel): # Interference model is the child class of simple model
   def __init__(self, penalty=0.09, **kwargs): # Passes all_other_arguments in the parent class to the child class. Then store penalty in an instance attribute. ** means that all arguments are packaged into this 'dictionary' 
      #print(kwargs) # Kwargs = Keyword arguments
      super().__init__(**kwargs) # Stars unfold the arguments in the dictionary
      self.penalty=penalty

      def simulate_trial(self, list_length): # Need to define the new trial simulation
         p_correct_recall, accurate_recall= super().simulate_trial(list_length=list_length)
         p_correct_recall -= self.penalty # Stores this calculation. = prob_correct_recall = penalty
         p_correct_recall=np.clip(p_correct_recall, 0.0, 1.0) # Look into what 'clipping does' -- I think this means it restricts the values by setting a boundary
         accurate_recall= True if self.rng.random() < p_correct_recall else False # ensure that the randomly generated value is less than p_correct_recall?

         return p_correct_recall, accurate_recall


if __name__=='__main__': # If the name assigned to the current module I'm running is main (needs to be specified from command line), then do the following...
    someones_memory = SimpleMemoryModel() # Individual 1's memory instance
    someone_elses_memory=InterferenceMemoryModel(penalty=0.3) # Individual 2's memory instance -- this participant's memory is worse. If this penalty wasn't directly specified here, it would've used the default value specified in line 32.

    #print(type(someone_elses_memory.rng))

    p1, acc1, = someones_memory.simulate_trial(['5', '7', '3', '4', '2']) # Treat each one as a string since these were examples of actual stimuli that were shown.
    print(f"Prob & Acc of recall for Individual 1:{p1}, {acc1}")

    p2, acc2 = someone_elses_memory.simulate_trial(['5', '7', '3', '4', '2'])
    print(f"Prob & Acc of recall for Individual 2:{p2}, {acc2}")

