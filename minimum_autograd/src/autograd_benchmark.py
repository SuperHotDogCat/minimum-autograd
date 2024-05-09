import numpy as np
from sklearn.datasets import make_moons
from minimum_autograd.assets import Tensor, ReLU, Sigmoid, BinaryCrossEntropyLoss

X,y = make_moons(n_samples=1000, random_state=0)
y = y.reshape(y.shape[0], -1)
x = Tensor(X, is_variable=False)
bceloss = BinaryCrossEntropyLoss()
label = Tensor(y, is_variable=False)

class Net:
    def __init__(self, input_size, hidden_size, n_classes = 1):
        self.W1 = Tensor(np.random.normal(size = (input_size, hidden_size)))
        self.W2 = Tensor(np.random.normal(size = (hidden_size, n_classes)))
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.parameters = [self.W1, self.W2]
    def __call__(self, x: Tensor):
        return self.forward(x)
    def forward(self, x: Tensor):
        x = x @ self.W1
        x = self.relu(x)
        x = x @ self.W2
        x = self.sigmoid(x)
        return x

class SGD:
    def __init__(self, parameters, lr = 1e-5):
        self.parameters = parameters
        self.lr = lr
    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0
        return
    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad
        self.lr /= 1.01
        return 

net = Net(2, 50, 1)
optimizer = SGD(net.parameters, 1e-2)

for i in range(1000):
    optimizer.zero_grad()
    pred = net(x)
    loss = bceloss(label, pred)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        pred = np.where(pred.data >= 0.5, 1, 0)
        print(f"Epoch: {i}")
        print("loss: ", loss.data, "acc: ", np.sum(pred == label.data) / len(label.data))