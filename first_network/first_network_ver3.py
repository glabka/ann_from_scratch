# bias
# training data
inputs = [1, 2, 3, 4]
targets = [x + 10 for x in [2, 4, 6, 8]]
epochs = 100
# slope value initialized by arbitrary value
w = 0.1 # weight
b = 0.3 # bias (initialized by arbitrary value)
learning_rate = 0.1

# the neuron
def predict(i):
    return w * i + b
# train the neuron (Batch gradient descent -> all data all used for updating weight(s) every iteration)
for _ in range(epochs):
    pred = [predict(i) for i in inputs] # feed forward: calculating the predicted values
    errors = [t - p for t, p in zip(targets, pred)]
    cost = sum(errors) / len(targets) # average error
    errors_s2_d = [2 * (p - t) for p, t in zip(pred, targets)] # derivative of squared errors
    weight_d = [e * i for i, e in zip(inputs, errors_s2_d)]
    bias_d = errors_s2_d # bias delta -> with respect to bias input which is always 1. Basically equal to [e * 1 for e in errors_s2_d].
    print(f"Weight: {w:.2f}, Bias: {b:.2f}, Cost: {cost:.2f}")
    w -= learning_rate * sum(weight_d) / len(weight_d) # back propagation: adjusting weight(s)s
    b -= learning_rate * sum(bias_d) / len(bias_d)

# test the network
test_inputs = [5, 6]
test_targets = [x + 10 for x in [10, 12]]
pred = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"input:{i}, target:{t}, pred:{p:.4f}")