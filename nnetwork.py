import numpy as np
from tqdm import tqdm

class nnetwork():

	def __init__(self, layers, activations, dF, dF2=[]):
		'''
			layers = [input_size, n_layer1, n_layer2, ..., n_layerN]
		'''
		self.W = [np.random.randn(layers[i], layers[i-1]) for i in range(1, len(layers))]
		self.b = [np.random.randn(layers[i], 1) for i in range(1, len(layers))]
		self.layers = layers
		self.activations = activations
		self.dF = dF
		self.d2F = []             
		self.a = [0]*(len(layers))
		
	def feedforward(self, x):
		a = x.reshape(-1, 1)
		self.a[0] = a
		i = 1
		for f, weight, bias in zip(self.activations, self.W, self.b):
			a = f(np.dot(weight, a) + bias)
			self.a[i] = a
			i += 1
		return a

	def error(self, current, target):
		return target - current

	def backpropagation(self, e):
		'''
			Retropropagacion usando gradiente descendiente
		'''
		S = [0]*(len(self.layers)-1)
		s = -2 * self.F(self.a[-1], self.dF[-1]) * e
		S[-1] = np.copy(s)
		for i in range(len(self.layers)-3, -1, -1):
			Fx = self.F(self.a[i+1], self.dF[i])
			Wt =  np.transpose(self.W[i+1])
			s = np.dot(Fx, np.dot(Wt, s))
			S[i] = np.copy(s)
		return S

	def SGD_update(self, sensitivities, learning_rate):
		for i in range(len(self.layers)-2, -1, -1):
			self.W[i] = self.W[i] - learning_rate * np.dot(sensitivities[i], np.transpose(self.a[i]))
			self.b[i] = self.b[i] - learning_rate * sensitivities[i]

	def momentum(self, e, p):
		'''
			Retropropagacion usando momento
		'''
		pass
	
	def levenberg_marquardt(self, e):
		pass

	def fit(self, X, Y, epochs, learning_rate):
		_error = []
		_epochs = []
		data = [(x, y) for x, y in zip(X, Y)]
		np.random.shuffle(data)
		for i in tqdm(range(epochs)):
			for x, y in data:
				output = self.feedforward(x)
				error = self.error(output, y)	
				sensitivities = self.backpropagation(error)	
				self.SGD_update(sensitivities, learning_rate)	
			_error.append(error[0,0])
			_epochs.append(i)

		return np.array(_error), np.array(_epochs)

	def predict(self, X):
		prediction = []
		for x in X:
			a = x.reshape(-1,1)
			for f, weight, bias in zip(self.activations, self.W, self.b):
				a = f(np.dot(weight, a) + bias)
			prediction.append(a)

		return np.array(prediction).flatten()


	def F(self, a, df):
		''' 
			Regresa la matriz diagonal de derivadas de primer orden
		'''
		if len(a.shape) > 1:
			return np.diag(df(a.reshape(-1)))
		else:
			return np.diag(df(a))