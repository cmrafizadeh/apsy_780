def clip01(prob): 
# Clip the probabilities between 0 & 1
    if prob > 1:
        prob = 1
    if prob < 0:
        prob = 0

    return prob