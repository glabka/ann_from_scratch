# linear regression network
# training data
inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]
# slope value initialized by arbitrary value
w = 0.1
learning_rate = 0.1

# the neuron
def predict(i):
    return w * i
# train the neuron (Batch gradient descent -> all data all used for updating weight(s) every iteration)
for _ in range(25):
    pred = [predict(i) for i in inputs] # feed forward: calculating the predicted values
    errors = [t - p for p, t in zip(pred, targets)]
    cost = sum(errors) / len(targets) # average error

    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")
    w += learning_rate * cost # back propagation: adjusting weight(s)s

# test the network
test_inputs = [5, 6]
test_targets = [10, 12]
pred = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"input:{i}, target:{t}, pred:{p:.4f}")