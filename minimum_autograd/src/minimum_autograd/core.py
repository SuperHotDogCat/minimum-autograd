from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Variable(ABC):
    def __init__(self, data: np.array) -> None:
        self.data = data
        self.grad = 0
        self.from_function = None

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
