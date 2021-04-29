import numpy as np


class NeuralNetwork:
    def __init__(self, hidden_sizes, activation_func, error_func, input_size=900, output_size=6, scale=1e-2):
        self.layer_num = len(hidden_sizes) + 1
        self.activation_func = activation_func
        self.error_func = error_func
        self.net = {}

        # get all dimensions in the network
        net_sizes = np.concatenate((input_size, hidden_sizes, output_size), axis=None)

        for i in range(self.layer_num):
            self.net['W_' + str(i + 1)] = scale * np.random.randn(net_sizes[i], net_sizes[i + 1])
            self.net['b_' + str(i + 1)] = np.zeros(net_sizes[i + 1])

    # Activation functions - Start
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)
        # return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def relu(self, z):
        return max(0, z)
    # Activation functions - End

    # Forward - Start
    def forward_pass(self, X):
        inputs = X
        self.caches = []

        for i in range(self.layer_num - 1):
            inputs, cache = self.activated_forward(inputs, self.net['W_' + str(i + 1)], self.net['b_' + str(i + 1)])
            self.caches.append(cache)

        scores, cache = self.forward(inputs, self.net['W_' + str(i + 2)], self.net['b_' + str(i + 2)])
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
