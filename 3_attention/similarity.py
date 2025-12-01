import numpy as np

class Similarity:
    def __init__(self, type='cosine'):
        self.type_of_similarity = type

    def _cosine(self, vec_a, vec_b):
        abdot = np.dot(vec_a, vec_b)
        length_a = np.linalg.norm(vec_a)
        length_b = np.linalg.norm(vec_b)
        cosine_ab = abdot / (length_a * length_b + 1e-9)
        print('I was called')
        return cosine_ab

    def _euclidean(self, vec_a, vec_b):
        return None

    def __call__(self, vec_a, vec_b):
        if self.type_of_similarity == 'cosine':
            dist = self._cosine(vec_a, vec_b)
        elif self.type_of_similarity == 'euclidean':
            dist = self._euclidean(vec_a, vec_b)

        return dist


if __name__ == '__main__':
    mysim = Similarity('cosine')
    mysim = Similarity('euclidean')