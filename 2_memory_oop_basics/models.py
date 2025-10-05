import numpy as np
from .helpers import clip01

class SimpleMemoryModel:
    def __init__(self, retrieval_decay=0.1, encoding_error=0.1, noise=0.05, seed=42): # Class constructor
        self.retrieval_decay = retrieval_decay  # Instance attributes: depends on the length of things stored in memory (list_length)
        self.encoding_error = encoding_error
        self.rng = np.random.default_rng(seed)
        self.noise = noise # this noise attribute determines the std-dev in prob of recall from trial to trial

    def simulate_trial(self, list_of_letters):
        list_length = len(list_of_letters)
        prob_correct_recall = (1 - self.encoding_error) - (self.retrieval_decay * list_length)
        prob_correct_recall = prob_correct_recall + self.rng.normal(0.0, self.noise)

        # Clip the probabilities between 0 & 1
        prob_correct_recall = clip01(prob_correct_recall)

        # Randomly draw a result based on the prob_correct_recall
        if prob_correct_recall > self.rng.random():
            accurate_recall = True
        else:
            accurate_recall = False

        return prob_correct_recall, accurate_recall


class InterferenceMemoryModel(SimpleMemoryModel):
    def __init__(self, penalty=0.09, **kwargs):
        # Pass all_other_arguments to parent class's constructor
        super().__init__(**kwargs)
        # Store penalty in an instance attribute
        self.penalty = penalty

    def simulate_trial(self, list_of_letters):
        p_correct_recall, accurate_recall = super().simulate_trial(list_of_letters=list_of_letters)
        p_correct_recall -= self.penalty
        p_correct_recall = np.clip(p_correct_recall, 0.0, 1.0)
        accurate_recall = True if self.rng.random() < p_correct_recall else False

        return p_correct_recall, accurate_recall



if __name__ == '__main__':
    someones_memory = SimpleMemoryModel() # Individual 1's memory instance
    someone_elses_memory = InterferenceMemoryModel(seed=32) # Individual 2's memory instance

    print("I'm the king of the world!!!")
    # print(type(someone_elses_memory.rng))

    p1, acc1 = someones_memory.simulate_trial(['5','7','3','4','2'])
    print(f"Prob & Acc of recall for Individual 1: {p1}, {acc1}")

    p2, acc2 = someone_elses_memory.simulate_trial(['5','7','3','4','2'])
    print(f"Prob & Acc of recall for Individual 2: {p2}, {acc2}")