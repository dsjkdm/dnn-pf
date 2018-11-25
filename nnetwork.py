import numpy as np

class nnetwork():
	def __init__(self, layers, activations):
		'''
			layers = [input_size, n_layer1, n_layer2, ..., n_layerN]
		'''
		self.W = [np.random.randn(layers[i], layers[i-1]) for i in range(1, len(layers))]
		self.b = [np.random.randn(layers[i]) for i in range(1, len(layers))]
		self.layers = layers
		self.activations = activations
		
	def feedforward(self, x):
		a = x
		for f, weight, bias in zip(self.activations, self.W, self.b):
			a = f(np.dot(weight, a) + bias)
		return a

	def error(self, current, target):
		return target - current

	def backpropagation(self):
		pass

	def sigmoid(x, derivative=False):
		if not derivative:
			return 1/()

	
	def fit(self, X, Y, epochs):
		for i in range(epochs):
			for x, y in zip(X, Y):
				output = self.feedforward(x)
				error = self.error(output, y)
				


	