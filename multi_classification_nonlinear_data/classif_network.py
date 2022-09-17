# three layer network (input, hidden layer, output)
import math

import flowers_data as data

# my "random" weights
# w_i_h = [[0.1, -0.2], [0.3, 0.24], [-0.12, -0.3], [0.11, 0.14]] # 4 hidden neurons
# w_h_o = [[0.2, 0.4, -0.18, -0.7], [-0.6, -0.31, -0.33, 0.7], [0.09, 0.89, -0.45, 0.7]] # 3 output neurons
# b_h = [0.6, -0.8, 0.9, 0.5] # 4 hidden neurons
# b_o = [0.1, -0.2, 0.4] # 3 output neurons

w_i_h = [[0.1, -0.2], [-0.3, 0.25], [0.12, 0.23], [-0.11, -0.22]]
w_h_o = [[0.2, 0.17, 0.3, -0.11], [0.3, -0.4, 0.5, -0.22], [0.12, 0.23, 0.15, 0.33]]
b_h = [0.2, 0.34, 0.21, 0.44]
b_o = [0.3, 0.29, 0.37]

epochs = 3300
learning_rate = 0.5 # for 0.2 error and my random weights -> act = [1.6105905391867527e-290, 1.0, 2.2652764749177764e-123], target = 1

def calculate_pred(weights, biases, inputs):
    return [[sum([w * i for w, i in zip(we, inp)]) + b for we,b in zip(weights, biases)] for inp in inputs]

def soft_max(predictions):
    m = max(predictions)
    tmp = [math.exp(p - m) for p in predictions]
    total = sum(tmp)
    return [t / total for t in tmp]

def leaky_relu(pred):
    if pred < 0:
        return pred / 3
    else:
        return pred

def leaky_relu_d(pred):
    return 1/3 if pred <= 0 else 1

def relu(pred):
    return max(0, pred)

def relu_d(pred):
    return 0 if pred <= 0 else 1

def activation(pred):
    return relu(pred)

def activationd_d(pred):
    return relu_d(pred)

def log_loss(act, target):
    # print(f"act: {act}, target: {target}")
    return -target * math.log(act) - (1 - target) * math.log(1 - act)

for epoch in range(epochs):
    pred_h = calculate_pred(w_i_h, b_h, data.inputs)
    act_h = [[activation(p) for p in pr] for pr in pred_h]
    pred_o = calculate_pred(w_h_o, b_o, act_h)
    act_o = [soft_max(p) for p in pred_o]
    # print(f"act_o: {act_o}")
    # print(f"min(flatten(act_o)): {min([item for sublist in act_o for item in sublist])}")

    # cost
    cost = sum([sum([log_loss(a, t) for a, t in zip(ac, tg)]) for ac, tg in zip(act_o, data.targets)]) / len(data.targets)
    print(f"epoch: {epoch} cost: {cost:.4f}")

    # error derivatives
    error_o_d = [[a - t for a, t in zip(ac, tg)] for ac, tg in zip(act_o, data.targets)]
    w_h_o_T = list(zip(*w_h_o))
    error_d_h = [[sum([w * d for w, d in zip(w_r, err_d)]) * activationd_d(p) for w_r, p in zip(w_h_o_T, pr)] for err_d, pr in zip(error_o_d, pred_h)] # error_d_h = w_h_o_T matrix multiplying errors_o_d elementwise multiplied with ReLU_d

    # Gradient hidden -> output
    act_h_T = list(zip(*act_h)) # act_h is input into output layer
    error_o_d_T = list(zip(*error_o_d))
    w_h_o_d_T = [[sum([d * a for d, a in zip(err_d, ac)]) for err_d in error_o_d_T] for ac in act_h_T]
    b_o_d = [sum([d for d in deltas]) for deltas in error_o_d_T]

    # Gradient input -> hidden
    error_d_h_T = list(zip(*error_d_h))
    w_i_h_d_T = [[sum([d * i for d, i in zip(deltas, inp)]) for deltas in error_d_h_T] for inp in list(zip(*data.inputs))]
    b_h_d = [sum([d for d in deltas]) for deltas in error_d_h_T]

    # update weights and biases or all layers
    w_h_o_d = list(zip(*w_h_o_d_T))
    for i in range(len(w_h_o_d)):
        for j in range(len(w_h_o_d[i])):
            w_h_o[i][j] -= learning_rate * w_h_o_d[i][j] / len(data.inputs)
        b_o[i] -= learning_rate * b_o_d[i] / len(data.inputs)
    w_i_h_d = list(zip(*w_i_h_d_T))
    for i in range(len(w_i_h_d)):
        for j in range(len(w_i_h_d[i])):
            w_i_h[i][j] -= learning_rate * w_i_h_d[i][j] / len(data.inputs)
        b_h[i] -= learning_rate * b_h_d[i] / len(data.inputs)


# testing
pred_h = calculate_pred(w_i_h, b_h, data.test_inputs)
act_h = [[activation(p) for p in pr] for pr in pred_h]
pred_o = calculate_pred(w_h_o, b_o, act_h)
act_o = [soft_max(pr) for pr in pred_o]

counter = 0
index = 0
for pred, target in zip(act_o, data.test_targets):
    correct = pred.index(max(pred)) == target.index(max(target))
    if correct:
        counter += 1
    print(f"i: {index}:{correct} => pred: {pred}, target: {target}")
    index += 1

print(f"{counter}/{len(data.test_targets)} = {100*counter/len(data.test_inputs)} %")