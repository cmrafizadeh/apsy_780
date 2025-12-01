import numpy as np

class WordEncoder:
    def __init__(self, dim=10):
        self.mydim = dim

    def encode(self, tbe):
        # This method takes a string and encodes that as a vector
        # in a high-dimensional space, on the unit hypersphere
        hash_tbe = abs(hash(tbe)) % 2**32
        rng = np.random.default_rng(hash_tbe) # Generate a rng using hash_tbe as seed
        ndvec = rng.normal(0.0, 1.0, self.mydim)

        # Normalise the vector so that it lies on the hypersphere
        ndvec_norm = ndvec / (np.linalg.norm(ndvec) + 1e-9)

        return ndvec_norm

# One-hot encoding
# APPLE -> [0, 0, 0, 0, 1]
# BREAD -> [0, 0, 0, 1, 0]

# 
# APPLE -> [0.1, 0.5, -0.3, 0.2, 0.8]
# BREAD -> [0.1, 0.4, 0.8, 0.9, -0.8]
# my_other_rng = np.random.default_rng(seed=42)
# encoding_of_apple = my_other_rng.normal(0.0, 1.0, 20)
# my_other_rng = np.random.default_rng(seed=43)
# encoding_of_bread = my_other_rng.normal(0.0, 1.0, 20)

if __name__ == "__main__":
    my_encoder = WordEncoder(20)

    words = ['Apple', 'Bread', 'Cloud']

    for word in words:
        encoding = my_encoder.encode(word)
        print(f'Here is an encoding for {word}: {encoding}')