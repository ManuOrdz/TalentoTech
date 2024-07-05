from Network import Dense,Relu,Softmax,CategoricalCrossentrotpy, SoftmaxCategoricalCrossentropy, SGD
from sklearn.datasets import make_moons
import numpy as np



X, y = make_moons()

dense1 = Dense(2,16)
activation1 = Relu()
dense2 = Dense(16,3)

loss_activation = SoftmaxCategoricalCrossentropy()
optimizer = SGD(learning_rate=0.05, decay=0.1, momentum=0.5)

for epoch in range(1001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    loss = loss_activation.forward(dense2.output,y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, '+
              f'accuracy: {accuracy}, ' +
              f'loss: {loss}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)