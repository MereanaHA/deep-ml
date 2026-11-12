import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    probabilities=[]
    for feature in features :
        z = sum(a*b for a,b in zip(feature,weights))+bias
        probabilities.append(1/(1+math.exp(-z)))
    mse = sum((t-p)**2 for t,p in zip(labels,probabilities))/len(labels)
    return probabilities, mse