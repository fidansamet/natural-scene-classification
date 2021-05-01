import ast
import math
import numpy as np


class NeuralNetwork:
    def __init__(self, hidden_sizes=None, activation_func=None, error_func=None, lr=None, model_name=None,
                 model_import=False, input_size=None, output_size=6):
        self.net = {}
        self.model_name = model_name

        if model_import:
            self.import_model()
        else:
            self.layer_num = len(hidden_sizes) + 1
            self.activation_func = activation_func
            self.error_func = error_func
            self.lr = lr
            self.output_size = output_size
            self.net = {}
            self.init_weights(input_size, hidden_sizes, output_size)

    def init_weights(self, input_size, hidden_sizes, output_size):
        # get all dimensions in the network
        net_sizes = np.concatenate((input_size, hidden_sizes, output_size), axis=None).astype(int)

        for i in range(self.layer_num):
            stdv = 1. / math.sqrt(net_sizes[i])
            self.net['w_' + str(i + 1)] = np.random.uniform(-stdv, stdv, (net_sizes[i], net_sizes[i + 1]))
            self.net['b_' + str(i + 1)] = np.random.uniform(-stdv, stdv, net_sizes[i + 1])

    # Activation functions - Start
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)
        # return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    # Activation functions and derivatives - End

    # Activation functions derivatives - Start
    # TODO
    def d_sigmoid(self, z):
        return z * (1 - z)

    # TODO
    def d_tanh(self, z):
        return 1 - z ** 2

    def d_relu(self, x, z):
        return (z > 0) * x

    # Activation functions derivatives - End

    def softmax(self, z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        z = np.sum(np.exp(shifted), axis=1, keepdims=True)
        log_probs = shifted - np.log(z)
        probs = np.exp(log_probs)
        return log_probs, probs

    def sum_neg_log_likelihood(self, z, y):
        log_probs, probs = self.softmax(z)
        N = z.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        d_x = probs.copy()
        d_x[np.arange(N), y] -= 1
        d_x /= N
        return loss, d_x

    # TODO
    def mean_sum_squared_err(self, z, y):
        log_probs, probs = self.softmax(z)
        n = z.shape[0]
        loss = np.sum(np.power((1 - log_probs[np.arange(n), y]), 2)) / n
        d_x = -1 * (2 * (1 - probs[np.arange(n), y])) / n
        return loss, d_x

    # Forward - Start
    def forward_pass(self, X, valid=False):
        inputs = X
        self.caches = []

        for i in range(self.layer_num - 1):
            inputs, cache = self.activated_forward(inputs, self.net['w_' + str(i + 1)], self.net['b_' + str(i + 1)])
            self.caches.append(cache)

        scores, cache = self.forward(inputs, self.net['w_' + str(self.layer_num)], self.net['b_' + str(self.layer_num)])
        if not valid:
            self.caches.append(cache)
        return scores

    def forward(self, x, w, b):
        z = x.reshape(x.shape[0], -1).dot(w) + b
        cache = (x, w, b)
        return z, cache

    def activate(self, z):
        if self.activation_func == 'sigmoid':
            activated = self.sigmoid(z)
        elif self.activation_func == 'tanh':
            activated = self.tanh(z)
        elif self.activation_func == 'relu':
            activated = self.relu(z)
        cache = z
        return activated, cache

    def activated_forward(self, x, w, b):
        z, fwd_cache = self.forward(x, w, b)
        activated, a_cache = self.activate(z)
        cache = (fwd_cache, a_cache)
        # 2nd arg cache?
        return activated, cache

    # Forward - End

    # Backward - Start
    def backward_pass(self, scores, y):
        gradients = {}
        loss, d_o = self.sum_neg_log_likelihood(scores, y)

        d_o, d_w, d_b = self.backward(d_o, self.caches.pop())
        gradients['w_' + str(self.layer_num)] = d_w
        gradients['b_' + str(self.layer_num)] = d_b

        for i in range(self.layer_num - 2, -1, -1):
            d_o, d_w, d_b = self.activated_backward(d_o, self.caches.pop())
            gradients['w_' + str(i + 1)] = d_w
            gradients['b_' + str(i + 1)] = d_b

        return loss, gradients

    def backward(self, d_o, cache):
        x, w, b = cache
        d_x = d_o.dot(w.T).reshape(x.shape)
        d_w = x.reshape(x.shape[0], -1).T.dot(d_o)
        d_b = np.sum(d_o, axis=0)
        return d_x, d_w, d_b

    def d_activate(self, d_o, x):
        if self.activation_func == 'sigmoid':
            d_x = self.d_sigmoid(x)
        elif self.activation_func == 'tanh':
            d_x = self.d_tanh(x)
        elif self.activation_func == 'relu':
            d_x = self.d_relu(d_o, x)
        return d_x

    def activated_backward(self, d_o, cache):
        fwd_cache, a_cache = cache
        d_a = self.d_activate(d_o, a_cache)
        return self.backward(d_a, fwd_cache)

    def update_weights(self, gradients):
        for param, w in self.net.items():
            updated_w = self.sgd(w, gradients[param])
            self.net[param] = updated_w

    def sgd(self, w, d_w):
        w -= self.lr * d_w
        return w

    # Backward - End

    def train(self, X, y):
        scores = self.forward_pass(X)
        loss, gradients = self.backward_pass(scores, y)
        return loss, gradients

    def predict(self, X):
        return self.forward_pass(X, valid=True)

    def extract_model(self):
        model_file = open(self.model_name, 'a')
        model_file.write(str(self.net))
        model_file.close()

    def import_model(self):
        model_file = open(self.model_name, "r")
        contents = model_file.read()
        self.net = ast.literal_eval(contents)
        model_file.close()
