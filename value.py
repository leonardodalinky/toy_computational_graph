from __future__ import annotations
import numbers
from typing import Optional, TYPE_CHECKING

from numpy import isin

if TYPE_CHECKING:
    from operation import Operation


class Value:
    def __init__(self, op: Optional[Operation]):
        self.op = op


class Scalar(Value):
    def __init__(self, value: numbers.Number, op: Optional[Operation] = None):
        super().__init__(op)
        self._value = float(value)

    def __add__(self, other):
        from operation import AddOperation
        if isinstance(other, Scalar):
            op = AddOperation()
            return op([self, other])
        elif isinstance(other, numbers.Number):
            op = AddOperation()
            return op([self, Scalar(other)])
        else:
            raise TypeError("unsupported type")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from operation import SubOperation
        if isinstance(other, Scalar):
            op = SubOperation()
            return op([self, other])
        elif isinstance(other, numbers.Number):
            op = SubOperation()
            return op([self, Scalar(other)])
        else:
            raise TypeError("unsupported type")

    def __rsub__(self, other):
        if isinstance(other, Scalar):
            return Scalar.__sub__(other, self)
        elif isinstance(other, numbers.Number):
            return Scalar.__sub__(Scalar(other), self)
        else:
            raise TypeError("unsupported type")

    def __mul__(self, other):
        from operation import MulOperation
        if isinstance(other, Scalar):
            op = MulOperation()
            return op([self, other])
        elif isinstance(other, numbers.Number):
            op = MulOperation()
            return op([self, Scalar(other)])
        else:
            raise TypeError("unsupported type")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from operation import DivOperation
        if isinstance(other, Scalar):
            op = DivOperation()
            return op([self, other])
        elif isinstance(other, numbers.Number):
            op = DivOperation()
            return op([self, Scalar(other)])
        else:
            raise TypeError("unsupported type")

    def __rtruediv__(self, other):
        if isinstance(other, Scalar):
            return Scalar.__truediv__(other, self)
        elif isinstance(other, numbers.Number):
            return Scalar.__truediv__(Scalar(other), self)
        else:
            raise TypeError("unsupported type")

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"Scalar({self._value})"

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        raise AttributeError("cannot set value of a scalar")
