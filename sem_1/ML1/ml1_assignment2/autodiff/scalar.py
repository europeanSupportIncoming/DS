import math


class Scalar:
    """Stores a single scalar value and its gradient."""

    def __init__(self, value, _children=(), _op=""):
        self.value = value
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.value + other.value, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.value * other.value, (self, other), "*")

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Scalar(self.value**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.value ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Scalar(0 if self.value < 0 else self.value, (self,), "ReLU")

        def _backward():
            self.grad += (out.value > 0) * out.grad

        out._backward = _backward

        return out

    def log(self):
        out = Scalar(math.log(self.value), (self,), "Log")

        def _backward():
            self.grad += (1 / self.value) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Scalar(math.exp(self.value), (self,), "Exp")

        def _backward():
            self.grad += (math.exp(self.value)) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        # topological order all of the children in the graph
        topological_order = []
        visited_nodes = set()

        def build_topological_order(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                for child in node._prev:
                    build_topological_order(child)
                topological_order.append(node)

        build_topological_order(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topological_order):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Scalar(value={self.value}, grad={self.grad})"
