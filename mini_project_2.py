import numpy as np
from .helpers import rand_letters

class SimpleMemoryModel:
    def __init__(self, encoding_error=0.1, retrieval_decay=0.05, noise=0.05, seed=None): # Class constructor
        self.retrieval_decay=retrieval_decay # Instance attributes: decay depends on the length of things stored in memory (list_length)
        self.encoding_error=encoding_error
        self.rng=np.random.default_rng(seed)
        self.noise=noise # Memory also has a "noise" attribute that determines the standard deviation in the probability of recall from trial to trial.
        self.memory=[]

    def store(self, list_length):
        self.memory = [0 for item in range(list_length)]
        p_store=1
        for item in range(list_length):
            p_store-=(self.encoding_error * np.random.rand())
            if np.random.rand()<p_store:
                self.memory[item]=1
        
        return self.memory

    def retrieve(self, list_length):
        p_retrieve=1
        for item in range(list_length):
            p_retrieve-=self.retrieval_decay
            if self.memory[item]==1:
                if np.random.rand()>p_retrieve:
                    p_retrieve-=np.random.rand() * self.retrieval_decay
                    self.memory[item]=0

        return sum(self.memory)/list_length
    

    def simulate_trial(self, list_length): 
       to_store=rand_letters(list_length)
       stored=self.store(list_length)
       retrieved=self.retrieve(list_length)
       accuracy=1 if sum(self.memory) == list_length else 0

       return retrieved, accuracy, to_store, stored

 
class InterferenceMemoryModel(SimpleMemoryModel): # Interference model is the child class of simple model
   def __init__(self, penalty=0.09, **kwargs): # Passes all_other_arguments in the parent class to the child class. Then store penalty in an instance attribute. ** means that all arguments are packaged into this 'dictionary' 
      #print(kwargs) # Kwargs = Keyword arguments
      super().__init__(**kwargs) # Stars unfold the arguments in the dictionary
      self.penalty=penalty

   def simulate_trial(self, list_length): 
       to_store=rand_letters(list_length)
       stored=self.store(list_length)
       retrieved=self.retrieve(list_length)
       accuracy=1 if sum(self.memory) == list_length else 0

       return retrieved, accuracy, to_store, stored
   
#Run simple model trial for p1
p1=SimpleMemoryModel()
print(p1.simulate_trial(list_length=4))

#Run interference model trial for p2
p2=InterferenceMemoryModel()
print(p2.simulate_trial(list_length=4))


