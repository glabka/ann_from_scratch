import math

from customizable_nn.custom_nn import CustomNN
from multi_classification_nonlinear_data import flowers_data


def relu(pred):
    return [max(0, p) for p in pred]

def relu_d(pred):
    return 0 if pred <= 0 else 1

def soft_max(predictions):
    m = max(predictions)
    tmp = [math.exp(p - m) for p in predictions]
    total = sum(tmp)
    return [t / total for t in tmp]

def log_loss(act, target):
    return -target * math.log(act) - (1 - target) * math.log(1 - act)

def cost_fun(act_o_l, targets_l):
    return sum([sum([log_loss(a, t) for a, t in zip(act, tg)]) for act, tg in zip(act_o_l, targets_l)]) / len(targets_l)

def diff(a, t):
    return a - t


nn = CustomNN(0.5, cost_fun, [2, 4, 3], [relu, soft_max], [relu_d, diff])
epochs = 3300
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