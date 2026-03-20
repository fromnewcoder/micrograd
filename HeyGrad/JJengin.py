class Value:
    def __init__(self, value, children=(), op=""):
        self.data = value
        self.prev = children
        self._op = op
        self.grad = 0
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other), op="+")

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        self._backward = backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other), op="*")

        def backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        self._backward = backward

        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        return f"Value({self.data})"

    def backward(self):

        topo = []
        visited = set()

        def buildTopo(v):
            if v not in visited:
                visited.add(v)
                for c in v.prev:
                    buildTopo(c)
                topo.append(v)
        buildTopo(self)

        self.grad = 1

        for v in reversed(topo):
            v._backward()
