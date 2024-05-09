
import numpy as np
from minimum_autograd.core import Function, Variable
"""
Interface:
class Variable(ABC):
    def __init__(self, data: np.array) -> None:
        self.data = data
        self.grad = None
        self.from_functions = []
    
    @abstractmethod
    def backward(self) -> None:
        pass
        
class Function(ABC):
    def __init__(self) -> None:
        self.inputs = []
        self.outputs = []
        self.grads = [] # 前の関数から受け取る, zero gradで0に

    @abstractmethod
    def forward(self, *args):
        # inputとoutputを勾配計算のために保存する
        pass

    @abstractmethod
    def backward(self) -> None:
        # 前のgradを頼りにgrad計算してvariable classのbackwardを実行する
        pass
"""

class ReLU(Function):
    def __init__(self):
        self.inputs = [0]
        self.outputs = [0]
        self.grad = 0 # 前の関数から受け取る
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        self.inputs[0] = x
        self.outputs[0] = Tensor(np.where(x.data > 0, x.data, 0))
        self.outputs[0].from_function = self
        return self.outputs[0]
    def backward(self) -> None:
        for input_data in self.inputs:
            input_data.grad += np.where(input_data.data > 0, 1, 0) * self.outputs[0].grad
            input_data.backward()
        return 

class Sigmoid(Function):
    def __init__(self):
        self.inputs = [0]
        self.outputs = [0]
        self.grad = 0 # 前の関数から受け取る
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        self.inputs[0] = x
        sigmoid = 1 / (1 + np.exp(-x.data))
        self.outputs[0] = Tensor(sigmoid)
        self.outputs[0].from_function = self
        return self.outputs[0]
    def backward(self) -> None:
        for input_data in self.inputs:
            input_data.grad += (1-self.outputs[0].data) * self.outputs[0].data * self.outputs[0].grad
            input_data.backward()
        return 

class Matmul(Function):
    def __init__(self):
        self.inputs = [0,0]
        self.outputs = [0]
        self.grad = 0 # 前の関数から受け取る
    def __call__(self, x: Variable, W: Variable):
        return self.forward(x, W)
    def forward(self, x: Variable, W: Variable):
        self.inputs[0] = x
        self.inputs[1] = W
        output = x.data @ W.data
        self.outputs[0] = Tensor(output)
        self.outputs[0].from_function = self
        return self.outputs[0]
    def backward(self) -> None:
        prev_grad = self.outputs[0].grad # どうせ1要素
        if self.inputs[0].is_variable and isinstance(prev_grad, np.ndarray):
            self.inputs[0].grad += prev_grad @ self.inputs[1].data.T
            self.inputs[0].backward()
        if self.inputs[1].is_variable and isinstance(prev_grad, np.ndarray):
            self.inputs[1].grad += self.inputs[0].data.T @ prev_grad
            self.inputs[1].backward()
        return 

class BinaryCrossEntropyLoss(Function):
    def __init__(self):
        self.inputs = [0, 0]
        self.outputs = [0]
        self.grad = 0 # 前の関数から受け取る
    def __call__(self, ground_truth, pred):
        return self.forward(ground_truth, pred)
    def forward(self, ground_truth, pred):
        self.inputs[0] = ground_truth
        self.inputs[1] = pred
        log_y = np.log(np.clip(pred.data, 1e-30, 1-1e-30))
        log_1_y = np.log(np.clip(1-pred.data, 1e-30, 1-1e-30))
        output = -ground_truth.data*log_y-(1-ground_truth.data)*log_1_y
        self.outputs[0] = Tensor(output.mean())
        self.outputs[0].from_function = self
        return self.outputs[0]
    def backward(self) -> None:
        ground_truth = self.inputs[0].data
        pred = self.inputs[1].data
        grad = (pred - ground_truth) / np.clip(ground_truth,1e-30, 1-1e-30) / np.clip(1-ground_truth, 1e-30, 1-1e-30)
        grad = np.clip(grad, -1, 1)
        self.inputs[1].grad += grad
        self.inputs[1].backward()
        return 


class Tensor(Variable):
    def __init__(self, data: np.array, is_variable: bool = True) -> None:
        super().__init__(data)
        self.is_variable = is_variable
    def __matmul__(self, W: Variable):
        self.matmul = Matmul()
        return self.matmul(self, W)
    def backward(self) -> None:
        if self.from_function:
            self.from_function.grad += self.grad # 一応
            self.from_function.backward()
        return 