import random

class CustomNN:


    # cost_fun - input predictions, targets
    # layers_dimensions contains dimensions of and output layer as well as all hidden layers. Order is from input to output layer.
    # layers_act_funcs - can be None, they are meant to work on the whole output of a layer. Input doesn't have act_fun
    # delta_calc_funs - last is func for last layer that take as argument predicted activation value and corresponding target value. Input doesn't have delta_calc or act_d funcs
    # prior funcs are derivations of what is summed as cost function (or something corresponding to that)
    def __init__(self, learning_rate, cost_fun, layers_dimensions, layers_act_funcs, delta_calc_funcs):
        self.weights_list = []
        self.learning_rate = learning_rate
        self.cost_fun = cost_fun
        self.layers_act_funcs = layers_act_funcs
        self.delta_calc_funcs = delta_calc_funcs

        for i in range(1, len(layers_dimensions)):
            input_dim = layers_dimensions[i - 1]
            output_dim = layers_dimensions[i]
            # self.weights_list.append([[random.random() - 0.5 for _ in range(input_dim)] for _ in range(output_dim)]) # TODO uncomment
            self.weights_list = [[[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[-3, 0, 0], [0, -3, 0], [0, 0, -3]]] # TODO -delete -  debug
            self.weights_list = [[[1, 0], [0, 1], [1, 1]], [[2, 0, 0], [0, -1, 0], [0, 0, 0]]]# TODO -delete -  debug

    def mini_batch_learn(self, inputs, targets_l):
        preds_l, act_o_l = list(zip(*[self.calc_preds(input) for input in inputs])) # _l as list for each input
        cost = self.cost_fun(act_o_l, targets_l)
        print(f"cost: {cost}")
        print(f"len(preds_l) = len(batches) : {len(preds_l)}, len(preds_l[0] = len(numOfLayers) - 1: {len(preds_l[0])}, preds_l: {preds_l}")
        errors_d_l = [self.calc_error_d(preds, targets) for preds, targets in zip(preds_l, targets_l)]
        print(f"len(errors_d_l) = len(batch): {len(errors_d_l)}, len(errors_d[0]) = len(layers) - 1: {len(errors_d_l[0])}, errors_d_l: {errors_d_l}")
        # gradients - list for every batch' input
        gradients_l = [gradients(errors_d_l, preds_l.inset(0, inputs))]


    def gradients(self, errors_d_l, preds_and_inp):
        return None # TODO


    # inputs - column vector.
    # return tuple contining list of prediction and of all layers as column vectors and list of activations from last layer
    def calc_preds(self, input):
        preds = []
        acts = []
        print(f"len(weights_list): {len(self.weights_list)}")
        for i in range(len(self.weights_list)):
            if i == 0:
                print(f"input: {input}, weights: {self.weights_list[i]}")
                preds.append(self.matrix_mult_vector(self.weights_list[i], input))
            else :
                print(f"input: {preds[i - 1]}, weights: {self.weights_list[i]}")
                preds.append(self.matrix_mult_vector(self.weights_list[i], acts[i - 1]))
            acts.append(self.layers_act_funcs[i](preds[i]))
        return preds, acts[-1]

    # a x b
    # A - column vector -> represented by list of numbers
    # b - row vector -> represented by list of numbers
    # returns matrix (list of lists of numbers)
    @staticmethod
    def vector_mat_mult_vector(a, b):
        tmp_dbg = [[a_num * b_num for b_num in b] for a_num in a]
        return tmp_dbg

    # A x b
    # A - matrix - has to be a list of lists of numbers
    # b - column vector - represented by list of numbers
    # returns list of numbers
    @staticmethod
    def matrix_mult_vector(a, b):
        tmp_dbg = [sum([a_num * b_num for a_num, b_num in zip(a_row, b)]) for a_row in a]
        return tmp_dbg

    # A x B
    # A - matrix (both dimensions are greater than 1)
    # B - matrix (both dimensions are greater than 1)
    # returns matrix
    @staticmethod
    def matrix_multiplication(a, b):
        tmp_dbg = [[sum([a_num * b_num for a_num, b_num in zip(a_row, b_column)]) for a_row in a] for b_column in CustomNN.transpose(b)]
        return CustomNN.transpose(tmp_dbg)


    # deep copy basically (doesn't affect original matrix)
    @staticmethod
    def transpose(m):
        print(f"transposing: {m}")
        return list(zip(*m))
        # return [list(a) for a in zip(*m)]

    # preds - list of predictions of every layer
    # targets - list of targets
    # returns list of column vectors representing errors derivatives/deltas for every layer with exception of the first
    def calc_error_d(self, preds, targets):
        errors_d = []
        for i in range(len(preds) - 1, -1, -1):
            if i == len(preds) - 1: # last layer
                error_o_d = [self.delta_calc_funcs[i](a, t) for a, t in zip(self.layers_act_funcs[i](preds[i]), targets)]
                errors_d.append(error_o_d)
            else:
                back_error = CustomNN.matrix_mult_vector(CustomNN.transpose(self.weights_list[i + 1]), errors_d[-1]) # "back error"
                errors_d_h = [err * self.delta_calc_funcs[i](p) for err, p in zip(back_error, preds[i])]
                errors_d.insert(0, errors_d_h)
        return errors_d
