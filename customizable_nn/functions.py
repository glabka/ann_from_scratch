import math

def relu(pred):
    return [max(0, p) for p in pred]

def relu_d(pred):
    return 0 if pred <= 0 else 1

def leaky_relu(pred):
    return [p / 3 if p < 0 else p for p in pred]

def leaky_relu_d(pred):
    return 1/3 if pred < 0 else 1

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
