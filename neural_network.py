import math
import pickle
import numpy as np


class NeuralNetwork:
    def __init__(self, hidden_sizes=None, activation_func=None, error_func=None, lr=None, model_import=False,
                 model_path=None, input_size=None, output_size=6):
        self.net = {}

        if model_import:
            self.model_path = model_path
            self.import_model()
        else:
            self.layer_num = len(hidden_sizes) + 1
            self.activation_func = activation_func
            self.error_func = error_func
            self.lr = lr
            self.output_size = output_size
            self.init_weights(input_size, hidden_sizes, output_size)

    def init_weights(self, input_size, hidden_sizes, output_size):
        # get all dimensions in the network
        layer_sizes = np.concatenate((input_size, hidden_sizes, output_size), axis=None).astype(int)

        for i in range(self.layer_num):
            stdv = 1. / math.sqrt(layer_sizes[i])
            self.net['w_' + str(i + 1)] = np.random.uniform(-stdv, stdv, (layer_sizes[i], layer_sizes[i + 1])).astype(
                'float32')
            self.net['b_' + str(i + 1)] = np.random.uniform(-stdv, stdv, layer_sizes[i + 1]).astype('float32')

    # Activation functions - Start
    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z))).astype('float32')

    def tanh(self, z):
        return np.tanh(z).astype('float32')
        # return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z).astype('float32')

    # Activation functions and derivatives - End

    # Activation functions derivatives - Start
    def d_sigmoid(self, a):
        return a * (1 - a)

    def d_tanh(self, a):
        return 1 - a ** 2

    def d_relu(self, z):
        return z > 0

    # Activation functions derivatives - End

    def softmax(self, z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        z = np.sum(np.exp(shifted), axis=1, keepdims=True)
        log_probs = shifted - np.log(z)
        probs = np.exp(log_probs)
        return log_probs, probs

    # Error functions - Start
    def sum_neg_log_likelihood(self, z, y):
        log_probs, probs = self.softmax(z)
        n = z.shape[0]
        loss = -np.sum(log_probs[np.arange(n), y]) / n
        d_x = probs.copy()
        d_x[np.arange(n), y] -= 1
        d_x /= n
        return loss, d_x

    def mean_squared_err(self, z, y):
        _, probs = self.softmax(z)
        n = z.shape[0]
        one_hot_y = np.zeros((n, self.output_size), dtype='float32')
        one_hot_y[np.arange(n), y] = 1.
        loss = np.sum(np.power(one_hot_y - probs, 2)) / n
        d_x = -2 * (one_hot_y - probs) / n
        return loss, d_x

    def sum_squared_err(self, z, y):
        _, probs = self.softmax(z)
        n = z.shape[0]
        one_hot_y = np.zeros((n, self.output_size), dtype='float32')
        one_hot_y[np.arange(n), y] = 1.
        loss = np.sum(np.power(one_hot_y - probs, 2))
        d_x = -2 * (one_hot_y - probs)
        return loss, d_x

    # Error functions - End

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
        return activated

    def activated_forward(self, x, w, b):
        z, fwd_cache = self.forward(x, w, b)
        activated = self.activate(z)
        return activated, (fwd_cache, z, activated)

    # Forward - End

    # Backward - Start
    def backward_pass(self, scores, y):
        gradients = {}
        if self.error_func == 'log':
            loss, d_o = self.sum_neg_log_likelihood(scores, y)
        elif self.error_func == 'sse':
            loss, d_o = self.sum_squared_err(scores, y)
        elif self.error_func == 'mse':
            loss, d_o = self.mean_squared_err(scores, y)

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

    def d_activate(self, d_o, z, a):
        if self.activation_func == 'sigmoid':
            d_x = self.d_sigmoid(a)
        elif self.activation_func == 'tanh':
            d_x = self.d_tanh(a)
        elif self.activation_func == 'relu':
            d_x = self.d_relu(z)
        return d_x * d_o

    def activated_backward(self, d_o, cache):
        fwd_cache, z_cache, a_cache = cache
        d_a = self.d_activate(d_o, z_cache, a_cache)
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
        name = '%dnn_lr=%0.3f_err=%s_act=%s_vgg.pkl' % (self.layer_num, self.lr, self.error_func, self.activation_func)
        model = {
            'layer_num': self.layer_num,
            'activation_func': self.activation_func,
            'net': self.net
        }
        model_file = open('./model/' + name, 'wb')
        pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    def import_model(self):
        model_file = open(self.model_path, 'rb')
        model = pickle.load(model_file)
        self.layer_num = model['layer_num']
        self.activation_func = model['activation_func']
        self.net = model['net']
