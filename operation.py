from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from value import Scalar


class Operation(ABC):
    def __init__(self):
        super().__init__()
        self._ctx: Optional[Dict] = None

    def __call__(self, *args) -> Scalar:
        self._ctx = dict()
        ret = self.forward(*args, ctx=self._ctx)
        ret.op = self
        return ret

    @staticmethod
    @abstractmethod
    def forward(self, inputs: List[Scalar], ctx=None) -> Scalar:
        pass

    @staticmethod
    @abstractmethod
    def backward(self, grad_output: float, ctx=None) -> List[float]:
        pass


class AddOperation(Operation):
    @staticmethod
    def forward(inputs: List[Scalar], ctx=None) -> Scalar:
        x, y = inputs
        return Scalar(x.value + y.value)

    @staticmethod
    def backward(grad_output: float, ctx=None) -> List[float]:
        return [grad_output, grad_output]


class SubOperation(Operation):
    @staticmethod
    def forward(inputs: List[Scalar], ctx=None) -> Scalar:
        x, y = inputs
        return Scalar(x.value - y.value)

    @staticmethod
    def backward(grad_output: float, ctx=None) -> List[float]:
        return [grad_output, -grad_output]


class MulOperation(Operation):
    @staticmethod
    def forward(inputs: List[Scalar], ctx=None) -> Scalar:
        x, y = inputs
        ctx["x"] = x.value
        ctx["y"] = y.value
        return Scalar(x.value * y.value)

    @staticmethod
    def backward(grad_output: float, ctx=None) -> List[float]:
        x, y = ctx["x"], ctx["y"]
        return [grad_output * y, grad_output * x]


class DivOperation(Operation):
    @staticmethod
    def forward(inputs: List[Scalar], ctx=None) -> Scalar:
        x, y = inputs
        assert y.value != 0, "Division by zero"
        ctx["x"] = x.value
        ctx["y"] = y.value
        return Scalar(x.value / y.value)

    @staticmethod
    def backward(grad_output: float, ctx=None) -> List[float]:
        x, y = ctx["x"], ctx["y"]
        return [grad_output / y, -grad_output / (y ** 2)]
