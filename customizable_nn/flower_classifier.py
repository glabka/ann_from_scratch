import math

from customizable_nn.custom_nn import CustomNN

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

def cost_fun(act_o, targets):
    return sum([sum([log_loss(a, t) for a, t in zip(act, tg)]) for act, tg in zip(act_o, targets)]) / len(targets)

def diff(a, t):
    return a - t

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

nn = CustomNN(0.1, cost_fun, [2, 4, 3], [relu, soft_max], [relu_d, diff])
print(nn.mini_batch_learn([[1, -1], [2, -2]], [[2, 1, 0], [4, 2, 0]]))

# nn = CustomNN(0, None, [3, 5, 2], [], [])
# print(nn.calc_preds([[1], [-1], [0]]))
# print(nn.calc_pred([[1, 2, 3], [2, 4, 8], [3, 6, 12]]))