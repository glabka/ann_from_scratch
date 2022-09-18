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

    def mini_batch_learn(self, inputs, targets):
        preds, act_o = self.calc_preds(inputs)
        cost = self.cost_fun(act_o, targets)
        print(f"cost: {cost}")
        print(f"len(preds): {len(preds)}, len(preds[0]): {len(preds[0])}, len(preds[0][0]): {len(preds[0][0])}, preds: {preds}")
        errors_d = self.calc_error_d(preds, targets)
        print(f"len(errors_d): {len(errors_d)}, len(errors_d[0]): {len(errors_d[0])}, len(errors_d[0][0]): {len(errors_d[0][0])}, preds: {preds}")
        return preds # debug

    # inputs - column vector.
    # return tuple contining list of prediction and of all layers as column vectors and list of activations from last layer
    def calc_preds(self, inputs):
        preds = []
        acts = []
        print(f"len(weights_list): {len(self.weights_list)}")
        for i in range(len(self.weights_list)):
            if i == 0:
                print(f"input: {inputs}, weights: {self.weights_list[i]}")
                preds.append(self.matrix_multiplication(self.weights_list[i], inputs))
            else :
                print(f"input: {preds[i - 1]}, weights: {self.weights_list[i]}")
                preds.append(self.matrix_multiplication(self.weights_list[i], acts[i - 1]))
            acts.append(self.layers_act_funcs[i](CustomNN.transpose(preds[i])[0])) # [0] beacause everything is a matrix so vector is nestead in extra list
        return preds, acts[-1]

    # A x B
    @staticmethod
    def matrix_multiplication(a, b):
        tmp_dbg = [[sum([a_num * b_num for a_num, b_num in zip(a_row, b_column)]) for a_row in a] for b_column in CustomNN.transpose(b)]
        return CustomNN.transpose(tmp_dbg)


    # deep copy basically (doesn't affect original matrix)
    @staticmethod
    def transpose(m):
        print(f"transposing: {m}")
        # if all(type(i) in [float, int] for i in m):
        #     return [[item] for item in m]
        # elif len(m) == 1 and isinstance(m[0], list):
        #     return [[[item] for item in m[0]]]
        # else:
        return [list(a) for a in zip(*m)]

    # preds - list of predictions of every layer
    # targets - list of targets
    # returns list of column vectors representing errors derivatives/deltas for every layer
    def calc_error_d(self, preds, targets):
        errors_d = []
        for i in range(len(preds) - 1, -1, -1):
            if i == len(preds): # last layer
                # errors_d.append()
                error_o_d = [[self.delta_calc_func[i](a, t) for a, t in zip(ac, tg)] for ac, tg in zip(self.layers_act_funcs[i](preds[i]), targets)]
                errors_d.append(error_o_d)
            else:
                # errors_d_i = [[sum([w * d for w, d in zip(w_r, err_d)]) * activationd_d(p) for w_r, p in zip(w_h_o_T, pr)] for err_d, pr in zip(error_o_d, pred_h)]
                # error_d_h = w_h_o_T matrix multiplying errors_o_d elementwise multiplied with ReLU_d
                errors_d_i = [[[[scal * der for scal in row] for der in self.delta_calc_funcs[i](p)] for p in preds[i]] for row in CustomNN.matrix_multiplication(CustomNN.transpose(self.weights_list[i + 1]), errors_d_i[i + 1])]
                errors_d.insert(0, errors_d_i)
        errors_d
