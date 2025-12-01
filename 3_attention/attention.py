import numpy as np
from .similarity import Similarity

class Attention:
    def __init__(self, type='softmax'):
        self.type = type

    def softmax(self, scores, temperature=1.0):
        # Scale scores based on temperature
        # Gaurav style
        # scores = [score_i / temperature for score_i in scores]

        # Anirudh style
        scores = np.array(scores)
        scores = scores / temperature

        # Exponentiate
        e_scores = np.exp(scores)

        # Normalise
        total = np.sum(e_scores)
        norm_e_scores = e_scores / total

        return norm_e_scores

class DotProductAttention:
    def __init__(self, temperature):
        self.temperature = temperature

    def _softmax(self, scores, temperature=1.0):
        # Scale scores based on temperature
        # Gaurav style
        # scores = [score_i / temperature for score_i in scores]

        # Anirudh style
        scores = np.array(scores)
        scores = scores / temperature

        # Exponentiate
        e_scores = np.exp(scores)

        # Normalise
        total = np.sum(e_scores)
        norm_e_scores = e_scores / total

        return norm_e_scores

    def attend(self, query, keys, values):
        # Compute similairity scores
        # sim = Similarity('cosine')
        # scores = []
        # for key in keys:
        #     sim_key = sim(query, key)
        #     scores.append(sim_key)
        scores = keys @ query

        # Convert scores to attention weights
        
        # my_attention = Attention()
        # weights = my_attention.softmax(scores)
        weights = self._softmax(scores, self.temperature)

        # Compute a context vector
        context = weights @ values

        return weights, context
