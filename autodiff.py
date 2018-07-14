import numpy as np

class Equation:
    var_dict = {}
    def __init__(self):
        self.var_dict = {}

    def __add__(self, exp2):
        return Add(self, exp2)

    def __sub__(self, exp2):
        return Subtract(self, exp2)
    
    def __mul__(self, exp2):
        return Multiply(self, exp2)
    
    def __truediv__(self, exp2):
        return Divide(self, exp2)

    def __pow__(self, exp2):
        return Power(self, exp2)

    def grad(self, var_dict):
        raise NotImplementedError
    
    def evaluate(self, var_dict):
        raise NotImplementedError

    def update(self, var_dict):
        Equation.var_dict.update(var_dict)

class Add(Equation):
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2

    def grad(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return self.exp1.grad() + self.exp2.grad()
    
    def evaluate(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return self.exp1.evaluate() + self.exp2.evaluate()


class Subtract(Equation):
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2

    def grad(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return self.exp1.grad() - self.exp2.grad()
    
    
    def evaluate(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return self.exp1.evaluate() - self.exp2.evaluate()


class Divide(Equation):
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2

    def grad(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return self.exp1.grad() / self.exp2.evaluate() - self.exp2.grad() * self.exp1.evaluate() / (self.exp2.evaluate() ** 2)
    
    
    def evaluate(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return self.exp1.evaluate()/self.exp2.evaluate()


class Multiply(Equation):
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2

    def grad(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return self.exp1.grad() * self.exp2.evaluate() + self.exp2.grad() * self.exp1.evaluate()
    
    def evaluate(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return self.exp1.evaluate() * self.exp2.evaluate()


class Power(Equation):
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2

    def grad(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return self.exp2.evaluate() * (self.exp1.evaluate() ** (self.exp2.evaluate() - 1))
    
    def evaluate(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return self.exp1.evaluate() ** self.exp2.evaluate()


class Constant(Equation):
    def __init__(self, _evaluate):
        self._evaluate = _evaluate
        
    def grad(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return 0

    def evaluate(self, var_dict={}):
        return self._evaluate


class Variable(Equation):
    def __init__(self, var_name):
        self.var_name = var_name
        
    def grad(self, var_dict={}):
        Equation.var_dict.update(var_dict)

        if isinstance(Equation.var_dict[self.var_name], int):
            return 1
        else:
            return np.ones_like(Equation.var_dict[self.var_name])
    
    def evaluate(self, var_dict={}):
        Equation.var_dict.update(var_dict)
        return self.var_dict[self.var_name]


class Function(Equation):
    class Sin(Equation):
        def __init__(self, var) :
            self.var = var

        def grad(self, var_dict={}):
            Equation.var_dict.update(var_dict)
            return np.cos(self.var.evaluate())

        def evaluate(self, var_dict={}):
            Equation.var_dict.update(var_dict)
            return np.sin(self.var.evaluate())
        
    class Cos(Equation):
        def __init__(self, var):
            self.var = var

        def grad(self, var_dict={}):
            Equation.var_dict.update(var_dict)
            return -np.sin(self.var.evaluate())

        def evaluate(self, var_dict={}):
            Equation.var_dict.update(var_dict)
            return np.cos(self.var.evaluate())
        
    class Tan(Equation):
        def __init__(self, var):
            self.var = var

        def grad(self, var_dict={}):
            Equation.var_dict.update(var_dict)
            return (Function.Sin(self.var)/Function.Cos(self.var)).grad(var_dict)

        def evaluate(self, var_dict={}):
            Equation.var_dict.update(var_dict)
            return np.tan(self.var.evaluate())   
        
    class Exp(Equation):
        def __init__(self, var):
            self.var = var

        def grad(self, var_dict={}):
            Equation.var_dict.update(var_dict)
            return np.exp(self.var.evaluate())

        def evaluate(self, var_dict={}):
            Equation.var_dict.update(var_dict)
            return np.exp(self.var.evaluate())


