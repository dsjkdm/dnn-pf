import numpy as np
from tqdm import tqdm

class network():

	def __init__(self, layers, activations, dF, dF2=[]):
		'''
			layers => [input_size, n_layer1, n_layer2, ..., n_layerN]
            vW  => velocity of weights
            vb  => velocity of biases
            dF  => 1st order derivative of activations
            d2F => 2nd order derivative of activations
            a   => output of each layer
            n   => output before the activation
		'''
		self.W = [np.random.randn(layers[i], layers[i-1])*np.sqrt(1/layers[i-1]) for i in range(1, len(layers))]
		self.b = [np.random.randn(layers[i], 1) for i in range(1, len(layers))]
		self.vW = [np.zeros((layers[i], layers[i-1])) for i in range(1, len(layers))]
		self.vb = [np.zeros((layers[i], 1)) for i in range(1, len(layers))]		
		self.layers = layers
		self.activations = activations
		self.dF = dF
		self.d2F = []
		self.dFx = [0]*(len(layers)-1)   
		self.H = [0]*(len(layers)-1)           
		self.a = [0]*(len(layers))
		self.n = [0]*(len(layers)-1)
		
	def feedforward(self, x):
		a = x.reshape(-1, 1)
		self.a[0] = a
		i = 1
		for f, weight, bias in zip(self.activations, self.W, self.b):
			n = np.dot(weight, a) + bias
			a = f(n)
			self.n[i-1] = n
			self.a[i] = a
			i += 1
		return a

	def error(self, current, target):
		return target - current

	def backpropagation(self, e, learning_rate, optimizer, beta, lmbda):
		'''
			Retropropagacion usando gradiente descendiente
		'''
		S = [0]*(len(self.layers)-1)

		Fx = self.F(self.a[-1], self.dF[-1])
		s = -2 * Fx * e
		self.dFx[-1] = np.copy(Fx)
		S[-1] = np.copy(s)
		for i in range(len(self.layers)-3, -1, -1):
			Fx = self.F(self.a[i+1], self.dF[i])
			Wt =  np.transpose(self.W[i+1])
			s = np.dot(Fx, np.dot(Wt, s))
			self.dFx[i] = np.copy(Fx)
			S[i] = np.copy(s)

		if optimizer == 'SGD':
			for i in range(len(self.layers)-2, -1, -1):
				self.W[i] = self.W[i] - learning_rate * np.dot(S[i], np.transpose(self.a[i]))
				self.b[i] = self.b[i] - learning_rate * S[i]

		if optimizer == 'momentum':
			for i in range(len(self.layers)-2):
				self.vW[i] = beta * self.vW[i] + learning_rate * np.dot(S[i], np.transpose(self.a[i]))
				self.vb[i] = beta * self.vb[i] + learning_rate * S[i]
				self.W[i] = self.W[i] - self.vW[i]
				self.b[i] = self.b[i] - self.vb[i]

		if optimizer == 'lma':
			J = [0]*(len(self.layers)-1)
			H = [0]*(len(self.layers)-1)
			for i in range(len(self.layers)-1):
				J[i] = np.dot(self.dFx[i], self.W[i])
				H[i] = np.dot(J[i], J[i].T)
				dW = np.dot(np.linalg.inv(H[i]-lmbda*np.diag(np.diag(H[i]))), np.dot(S[i], self.a[i].T))
				db = np.dot(np.linalg.inv(H[i]-lmbda*np.diag(np.diag(H[i]))), S[i])
				self.W[i] = self.W[i] - dW
				self.b[i] = self.b[i] - db


	def fit(self, X, Y, epochs, learning_rate, method, **kwargs):
		if kwargs:
			try:
				beta = kwargs['beta']
			except:
				beta = 0
			try:
				lmbda = kwargs['lmbda']
			except:
				lmbda = 0
			lmbda = kwargs['lmbda']
		else:
			beta = 0.0
			lmbda = 0.0
		_error = []
		_epochs = []
		data = [(x, y) for x, y in zip(X, Y)]
		np.random.shuffle(data)        
		for i in tqdm(range(epochs)):
			for x, y in data:
				output = self.feedforward(x)
				error = self.error(output, y)	
				self.backpropagation(error, learning_rate, method, beta, lmbda)	
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

	def score(self, X, Y):
		accuracy = 0
		prediction = self.threshold(self.predict(X), 0.5)
		for y, y_pred in zip(self.threshold(Y, 0.5), prediction):
			if y == y_pred:
				accuracy += 1
		return accuracy/len(Y) * 100

	def F(self, a, df):
		''' 
			Regresa la matriz diagonal de derivadas de primer orden
		'''
		return np.diag(df(a.reshape(-1)))
		#if len(a.shape) > 1:
		#	return np.diag(df(a.reshape(-1)))
		#else:
		#	return np.diag(df(a))

	def threshold(self, X, thresh):
		f = np.vectorize(lambda x: 0 if x<thresh else 1)
		return f(X)