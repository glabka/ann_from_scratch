# Cost function (error function / loss funtion)
* terminology - cost function (seems to be related to average over all samples) while loss or error function are denoting single training sample
* many types for different applications
# Multicolinearity
* inputs highly correlated -> network is not able to understand the influence of individual input. 
# Cleaning the data
* for instance dealing with empty data
# Normalization of data
* different nominal values of data can result in ratio of adjustments of one weight to other weight that is too big and therefore one is lets say too small.
# Linear regression vs Logistic regression
# Classification
* idea -> for binary class. only one output neuron with values form 0 to 1. For more categories -> output neuron for each of them.
* Certainties -> target values in classification (0 for not the thing as 0 %, 1 for the thing as 100 %)
* One hot encoding - for three classes, fist class is (1, 0, 0), second is (0, 1, 0), etc.
# dead neuron
* output is zero for all training samples (in classification)
* back propagation is not capable of reviving the neuron
* can happen due to ReLU
# Vanishing gradient problem 
* when value of gradients are too small because of derivative(s) calculated upon activation functions 
* the gradient values can be so small, that the weight change is lost in computational precision
* can happen for instance with derivative of sigmoid
# Handy functions
* sigmoid function: real numbers to <0, 1>
* derivative of sigmoid
* ReLu (Rectified Linear Unit) - 0 for x in (-inf, 0>, x for x in (0, inf)
* Leaky ReLu (for dealing with dead neuron). Smaller slope for (-inf, 0> then for (0, inf)