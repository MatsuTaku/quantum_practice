from blueqat.pauli import *
import itertools

class SequentialAdderExpr():
    def __init__(self):
        self.terms = defaultdict(complex)

    def add(self, val):
        if isinstance(val, Number):
            val = Expr.from_number(val)
        if isinstance(val, Term):
            val = val.to_expr()
        for ops, coeff in val:
            self.terms[ops] += coeff

    def expr(self):
        return Expr.from_terms_dict(self.terms)

class SequentialMultiplierExpr():
    def __init__(self):
        self.T = []
        self.count = 0

    def mul(self, term):
        n = self.count
        self.count += 1
        self.T.append(term)
        for i in range(64):
            if (n >> i) & 1:
                self.T[-2] = (self.T[-2] * self.T[-1]).simplify()
                self.T = self.T[:-1]
            else:
                break

    def expr(self):
        self.count = 0
        while len(self.T) > 1:
            self.T[-2] = (self.T[-2] * self.T[-1]).simplify()
            self.T = self.T[:-1]
        return self.T[0]

def addition_exprs(exprs, coeff=1.0):
    adder = SequentialAdderExpr()
    for expr in exprs:
        adder.add(expr)
    return adder.expr() * coeff if coeff != 1.0 else adder.expr()

def multiply_exprs(exprs, coeff=1.0):
    multiplier = SequentialMultiplierExpr()
    for expr in exprs:
        multiplier.mul(expr)
    return multiplier.expr() * coeff if coeff != 1.0 else multiplier.expr()

def efficient_pow2_expr(expr):
    expr = expr.simplify()
    terms = []
    for i in range(len(expr.terms)):
        for j in range(i, len(expr.terms)):
            if i == j:
                terms.append(Term.from_pauli(I, expr.terms[i].coeff**2))
            else:
                term = (expr.terms[i]*expr.terms[j]).simplify()
                terms.append(term*2)
    return Expr.from_terms_iter(iter(terms)).simplify()
