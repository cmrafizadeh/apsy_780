import numpy as np

# Create a new class called Coin that can be flipped
class Coin:
    """ A simple class for a coin. Has value of coin and sides.
        You can flip the coin
    """
    sides = ['heads', 'tails'] # class attributes

    def __init__(self, value=25, seed=10): # constructor
        self.value = value # instance attribute
        self.side_up = self.sides[0]
        self.rng = np.random.default_rng(seed)

    def __repr__(self):
        return f"I'm a Coin. My value is {self.value} and I'm facing {self.side_up} side up."

    def flip(self):
        self.side_up = self.rng.choice(self.sides)

x = 24
y = 25
print(x, type(x))
print(y, type(y))
pocket_coin1 = Coin(value = 10)
pocket_coin2 = Coin()

print(pocket_coin1, type(pocket_coin1))
print(pocket_coin2)

pocket_coin1.flip()
print(pocket_coin1)

pocket_coin2.flip()
print(pocket_coin2)