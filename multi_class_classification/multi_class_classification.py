# classification - three color of flowers, two parameters of the leafs
import math

import flowers_data

weights = [[0.1, 0.2], [0.15, 0.25], [0.18, 0.1]]
biases = [0.3, 0.4, 0.35]
epochs = 5000
learning_rate = 0.9

def softmax(predictions):
    m = max(predictions)
    tmp = [math.exp(p - m) for p in predictions] # this is done to combat overflow and underflow (also it make it always positive)
    total = sum(tmp)
    return [t / total for t in tmp] # positive values made to have their sum equal to one == 100 %

# for one neuron
def log_loss(act, target):
    return -target * math.log(act) - (1 - target) * math.log(1 - act)

# training loop
for epoch in range(epochs):
    pred = [[sum([w * i for w, i in zip(we, inp)]) + b for we, b in zip(weights, biases)] for inp in flowers_data.inputs]
    act = [softmax(p) for p in pred]
    # It's only divided by number of training samples but not a number of categories it's recognizing - might be harder
    # to compare loss functions between two neural network with different number of categorized classes but better for
    # computation
    cost = sum([sum([log_loss(a, t) for a, t in zip(ac, ta)]) for ac, ta in zip(act, flowers_data.targets)]) / len(act)
    print(f"epoch: {epoch}, cost: {cost:.4f}")
    error_d = [[(a - t) for t, a in zip(ta, ac)] for ta, ac in zip(flowers_data.targets, act)]
    error_d_T = list(zip(*error_d))
    weights_d_T = [[sum([e * i for e, i in zip(err, inp)]) / len(inp) for err in error_d_T] for inp in list(zip(*flowers_data.inputs))]
    weights_d = list(zip(*weights_d_T))
    biases_d = [sum(e) / len(e) for e in error_d_T]
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            weights[i][j] -= learning_rate * weights_d[i][j]
    for i in range(len(biases)):
        biases[i] -= learning_rate * biases_d[i]

# testing the network
pred = [[sum([w * i for w, i in zip(we, inp)]) + b for we, b in zip(weights, biases)] for inp in flowers_data.test_inputs]
act = [softmax(p) for p in pred]
counter = 0
for i in range(len(act)):
    correctness = act[i].index(max(act[i])) == flowers_data.test_targets[i].index(max(flowers_data.test_targets[i]))
    print(f"index: {i} ({correctness}): act = {act[i]}, target = {flowers_data.test_targets[i]}")
    if correctness:
        counter += 1
print(f"result = {counter}/{len(flowers_data.test_targets)} which is {100*counter/len(flowers_data.test_targets):.3f} %")