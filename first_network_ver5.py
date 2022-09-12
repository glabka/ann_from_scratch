# normalized data and more inputs, (gradient descent)
# normalized input
inputs = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
    (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
    (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
    (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290, 870, 1545, 1480, 1750, 1845, 1790, 1955]
epochs = 1000
# slope value initialized by arbitrary value
weights = [0.1, 0.2] # weight
b = 0.3 # bias (initialized by arbitrary value)
learning_rate = 0.4

# the neuron
def predict(inp):
    return sum([w * i for w, i in zip(weights, inp)]) + b
# train the neuron (Batch gradient descent -> all data all used for updating weight(s) every iteration)
for ep in range(epochs):
    pred = [predict(inp) for inp in inputs] # feed forward: calculating the predicted values
    errors = [t - p for t, p in zip(targets, pred)]
    cost = sum(errors) / len(targets) # average error
    errors_s2_d = [2 * (p - t) for p, t in zip(pred, targets)] # derivative of squared errors
    weights_d = [[e * i for i in inp] for inp, e in zip(inputs, errors_s2_d)] # MY NOTE: I think that this is no longer derivative of now changed weight function at least not partial
    bias_d = errors_s2_d # bias delta -> with respect to bias input which is always 1
    print(f"Epochs: {ep}, Cost: {cost:.2f}")
    weights_d_T = list(zip(*weights_d))# transposed matrix of weights_d

    for i in range(len(weights)):
        weights[i] -= learning_rate * sum(weights_d_T[i]) / len(weights_d_T[i])

    b -= learning_rate * sum(bias_d) / len(bias_d)

# test the network
test_inputs = [(0.1600, 0.1391), (0.5600, 0.3046),
    (0.7600, 0.8013), (0.9600, 0.3046), (0.1600, 0.7185)]
test_targets = [500, 850, 1650, 950, 1375]
pred = [predict(inp) for inp in test_inputs]
for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"input:{i}, target:{t}, pred:{p:.4f}")