from Network import Dense,Relu,Softmax, CategoricalCrossentrotpy, SGD
from sklearn.datasets import make_moons
import numpy as np

X, y = make_moons()

dense1 = Dense(2,64)
activation1 = Relu()
dense2 = Dense(64,3)
activation2 = Softmax()
loss_funcition = CategoricalCrossentrotpy()
optimizer = SGD()


dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

loss = loss_funcition.calulate(dense2.output,y)
print('loss: ',loss)

predictions = np.argmax(loss_funcition.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y,axis=1)
accuracy = np.mean(predictions==y)
print('acc: ', accuracy)


loss_funcition.backward(loss_funcition.output, y)
dense2.backward(loss_funcition.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

optimizer.update_params(dense1)
optimizer.update_params(dense2)