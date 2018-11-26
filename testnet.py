from nnetwork import nnetwork
from matplotlib import pyplot as plt
import numpy as np

sigmoid = np.vectorize(lambda x: 1/(1+np.power(np.e, -x)))
dxsigmoid = np.vectorize(lambda x: sigmoid(x) * (1 - sigmoid(x)))

tanh = np.vectorize(lambda x: (np.e**x-np.e**(-x))/(np.e**x+np.e**(-x)))
dxtanh = np.vectorize(lambda x: 1 - np.square(tanh(x)))

relu = np.vectorize(lambda x: 0 if x<0 else x)
dxrelu = np.vectorize(lambda x: 0 if x<0 else 1)

threshold = np.vectorize(lambda x: 0 if x<0.5 else 1)

activations  = [tanh, relu, sigmoid]
derivatives  = [dxtanh, dxrelu, dxsigmoid]
#derivatives2 = [dx2tanh, dx2relu, dx2sigmoid]

x1 = np.array([[-4,-3],[-4,-1],[-3,-3],[-3,-1],[-3,0],[-2,-2],[0,-1],[1,-1],[1,0],[1,2],
               [2,-1],[2,1],[2,2],[2,3],[2,4],[3,-4],[3,-3],[3,-2],[3,-1],[4,-1]])

x2 = np.array([[-1,-4],[-1,-3],[-1,2],[-1,3],[-1,4],[0,-4],[0,-3],[0,2],[1,-5],[1,-4],
               [1,4],[3,2],[3,3],[4,1],[4,2]])

X = np.concatenate((x1, x2))
y = np.concatenate((np.zeros(20), np.ones(15)))

net = nnetwork([2, 30, 30, 1], activations, derivatives)
# lr = 0.005
_error, _epochs = net.fit(X, y, epochs=200, learning_rate=0.05)

predicted = threshold(net.predict(x2))

print(predicted)

plt.figure(1)
plt.subplot(121)
plt.plot(_epochs, np.abs(_error))
plt.grid(True)
plt.title('Error')
plt.xlabel('epochs')
plt.ylabel('error')
plt.subplot(122)
plt.scatter(x1[:,0], x1[:,1], marker='+')
plt.scatter(x2[:,0], x2[:,1], marker='o')
plt.xticks(range(-8, 9))
plt.yticks(range(-8, 9))
plt.grid(True)
#plt.plot(predicted[:20], range(20))
plt.show()

steps = 250
xx, yy = np.meshgrid(np.linspace(-8, 8, steps), np.linspace(-8, 8, steps))
xxyy = np.hstack((xx.reshape(steps*steps, 1), yy.reshape(steps*steps, 1)))
grid_prediction = threshold(net.predict(xxyy)) 
grid_prediction.shape = (steps, steps)

plt.figure(2)
plt.contourf(xx, yy, grid_prediction, cmap='jet')
plt.scatter(x1[:,0], x1[:,1])
plt.scatter(x2[:,0], x2[:,1], marker='x')
plt.title('Superficie de desicion')
plt.show()