import math

from customizable_nn.custom_nn import CustomNN
from customizable_nn.functions import cost_fun, soft_max, relu_d, diff
from multi_classification_nonlinear_data import flowers_data
from multi_classification_nonlinear_data.classif_network import relu

nn = CustomNN(0.5, cost_fun, [2, 4, 3], [relu, soft_max], [relu_d, diff])
epochs = 3300
# nn = CustomNN(0.5, cost_fun, [2, 5, 3], [relu, soft_max], [relu_d, diff])
# epochs = 3300
# nn = CustomNN(0.25, cost_fun, [2, 4, 3, 4, 3], [leaky_relu, leaky_relu, leaky_relu, soft_max], [leaky_relu_d, leaky_relu_d, leaky_relu_d, diff])
# epochs = 1000
for epoch in range(epochs):
    print(f"--------------- epoch: {epoch} ---------------")
    nn.mini_batch_learn(flowers_data.inputs, flowers_data.targets)

# testing
predictions = [nn.get_pred(inp) for inp in flowers_data.test_inputs]
counter = 0
for p, tg in zip(predictions, flowers_data.test_targets):
    correct = p.index(max(p)) == tg.index(max(tg))
    print(f"correct: {correct}, pred = {p}, targets = {tg}")
    if correct:
        counter += 1
print(f"categorized {counter}/{len(flowers_data.test_targets)} which is {100*counter/len(flowers_data.test_targets)} %")

# testing

# m = [[1, 2], [3, 4]]
# print(zip(*m))
# print(list(zip(*m)))
# print(CustomNN.transpose([[0], [1], [2]]))
# print(CustomNN.transpose(CustomNN.transpose([[0], [1], [2]])))
# print(CustomNN.transpose([[1, 2, 3], [4, 5, 6]]))

# print(CustomNN.vector_mat_mult_vector([1, 2, 3, 4], [1, -1]))
# print(CustomNN.matrix_mult_vector([[1, 2, 3, 4]], [1, -1, 0, 0]))
# print(CustomNN.matrix_mult_vector([[1, 2, 3, 4], [10, 20, 30, 40]], [1, -1, 0, 0]))

# nn = CustomNN(0.1, cost_fun, [2, 4, 3], [None, relu, soft_max], [diff, relu_d, diff])

# print(nn.mini_batch_learn([[1, -1], [2, -2]], [[2, 1, 0], [4, 2, 0]]))
# print(nn.mini_batch_learn([[1, -1], [2, -2]], [[0, 1, 0], [0, 1, 0]]))



# nn = CustomNN(0, None, [3, 5, 2], [], [])
# print(nn.calc_preds([[1], [-1], [0]]))
# print(nn.calc_pred([[1, 2, 3], [2, 4, 8], [3, 6, 12]]))