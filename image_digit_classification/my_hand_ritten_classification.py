import mnist_data_reader as reader
from customizable_nn.custom_nn import CustomNN
from customizable_nn.functions import cost_fun, relu, soft_max, relu_d, diff

epochs = 3
batch_size = 2000

nn = CustomNN(0.2, cost_fun, [28*28, 50, 30, 10], [relu, relu, soft_max], [relu_d, relu_d, diff])
for epoch in range(epochs):
    print(f"--------------- epoch: {epoch} ---------------")
    counter = 0
    for labels, targets, inputs in reader.get_training_samples(batch_size):
        nn.mini_batch_learn(inputs, targets)
        counter += 1

# labels, targets, inputs = reader.get_test_samples()
# for v, i in zip(labels[:10], inputs[:10]):
#     print(v)
#     reader.plot_number(i)
#     print()

test_labels, test_targets, test_inputs = reader.get_test_samples()
predictions = [nn.get_pred(t_i) for t_i in test_inputs]
counter = 0
for p, tg in zip(predictions, test_targets):
    correct = p.index(max(p)) == tg.index(max(tg))
    print(f"correct: {correct}, pred = {p}, targets = {tg}")
    if correct:
        counter += 1
print(f"categorized {counter}/{len(test_targets)} which is {100*counter/len(test_targets)} %") # classifies correctly circa 43 % for 1 epoch, 67 % for 3 epochs

