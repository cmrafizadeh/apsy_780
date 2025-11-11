from .encoder import WordEncoder
from .similarity import Similarity
from .attention import Attention
from .attention import DotProductAttention
import numpy as np
import matplotlib.pyplot as plt

def p_recall_from_similarity(similarity, gain=5.0, bias=-2.0):
    x = gain * similarity + bias
    return 1.0 / (1.0 + np.exp(-x))

if __name__ == "__main__":

    my_encoder = WordEncoder(3)

    words = ['Apple', 'Bread', 'Cloud']
    cue = 'Cloud'

    # Encode all the words in the list called 'words'
    list_of_encodings = []
    for word in words:
        encoding = my_encoder.encode(word)
        print(f'Here is an encoding for {word}: {encoding}')
        list_of_encodings.append(encoding)

    # Encode cue as well and then get distance between cue & words in embedding space
    sim = Similarity('cosine')
    list_of_sims = []
    encoding_cue = my_encoder.encode(cue) # encode the cue
    for word_ix, word in enumerate(words):
        e = list_of_encodings[word_ix]
        sim_word = sim(encoding_cue, e) # get the distance
        list_of_sims.append(sim_word)

    print(f"Similarities to {cue}: {list_of_sims}")

    attention = Attention()
    weights = attention.softmax(list_of_sims, temperature=0.9)
    print(f"Attention weights for {cue}: {weights}")


    # Use dot product attention to calculate context
    dp_attention = DotProductAttention(0.5)
    weights, context = dp_attention.attend(encoding_cue, list_of_encodings, list_of_encodings)

    print(f"Attention weights: {weights}")
    print(f"Context: {context}")

    # Test delay between encoding and recall
    test_delays = np.arange(0, 50, 5)
    recall_results = []

    for delay in test_delays:
        # Simulate decay of list of encodings
        decay_factor = np.exp(-delay / 10.0)

        # Add gaussian noise to list of encodings scaled by decay factor
        list_of_encodings = [e + np.random.normal(0.0, 0.1, e.shape) * decay_factor for e in list_of_encodings]

        # Simulate recall
        weights, context = dp_attention.attend(encoding_cue, list_of_encodings, list_of_encodings)

        # Calculate similarity between context and encoding_cue
        similarity = sim(context, encoding_cue)

        # Convert similarity to probability of recall
        p_recall = p_recall_from_similarity(similarity)

        # Simulate recall
        recall_result = np.random.random() < p_recall
        recall_results.append(recall_result)

    # Plot results
    plt.plot(test_delays, recall_results)
    plt.xlabel('Delay (ms)')
    plt.ylabel('Recall probability')
    plt.title('Recall probability as a function of delay')
    plt.show()