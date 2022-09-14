# classification - three color of flowers, two parameters of the leafs

weights = [[0.1, 0.2], [0.15, 0.25], [0.18, 0.1]]
biases = [0.3, 0.4, 0.35]
epochs = 1
learning_rate = 0.1

# training loop
for epoch in range(epochs):
    pred = 