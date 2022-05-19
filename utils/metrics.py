import numpy as np

def embedding_learning_accuracy(pred, label):
    accu = 0.0

    pred = [1 if item>0.5 else 0 for item in pred]
    num_match = (np.array(pred) == np.array(label)).sum()
    accu = num_match / len(label)
    accu = round(accu, 4)

    return accu

