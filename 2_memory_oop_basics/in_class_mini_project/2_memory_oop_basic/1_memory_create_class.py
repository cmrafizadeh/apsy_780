import numpy as np

# Create a new class called "coin" that can be flipped
class Coin:
    """ A simple class for a coin. Has value of coin and sides.
    You can flip the coin
    """
    sides=['heads', 'tails'] # These are class attributes shared by all instances of the class

    def __init__(self, value=25, seed=42): # The constructor--'value' specified here is the default value if not otherwise specified. At this seed, it's randomly taking the value "heads". A seed allows you to replicate your code (key to debugging)
        self.value=value # 'self' value is an instance attribute
        self.side_up=self.sides[0] # Whichever is the first thing in the list will be side up
        self.rng=np.random.default_rng(seed)

    def __repr__(self):
        return f"I'm a Coin. My value is {self.value} and I'm facing {self.side_up} side up."

    def flip(self):
        self.side_up=self.rng.choice(self.sides)

pocket_coin1=Coin(value =10)
pocket_coin2=Coin()

print(pocket_coin1)
print(pocket_coin2)

# Flips coin 1
pocket_coin1.flip()
print(pocket_coin1)

# Flips coin 2
pocket_coin2.flip()
print(pocket_coin2)

#x is an instance of class "int" AKA integer. Could use any letter or string to refer to an integer
x=24
y=25
print(x, type(x))
print(y, type(y))


