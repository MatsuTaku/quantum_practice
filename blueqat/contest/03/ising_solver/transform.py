from blueqat.pauli import *
from openfermion.ops import QubitOperator
from .utils import *
from .exprtools import *

class ExtendedId:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.cap = n_qubits
        self.map = {}
        self.invmap = {}

    def __repr__(self):
        map_str = ['{0} -> {1}'.format(map, self.map[map]) for map in self.map]
        joined = ',\n'.join(map_str)
        return joined

    def id(self, a, b):
        if a > b:
            a, b = b, a
        key = (a, b)
        if key not in self.map:
            cap = self.cap
            self.cap += 1
            self.map[key] = cap
            self.invmap[cap] = key
        return self.map[key]

    def src(self, n):
        if n < self.n:
            return n
        if n not in self.invmap:
            return None
        return self.invmap[n]

def _locality_reduced_expr(term, extended_id):
    if not isinstance(term, Term):
        return NotImplemented

    if len(term.ops) <= 2:
        return term.to_expr()

    sp1 = term.ops[0]
    sp2 = term.ops[1]
    zchain = Term.from_ops_iter(term.ops[2:], 1.0)
    id1, id2 = 0, 0
    for i, n in enumerate(term.n_iter()):
        if i == 0:
            id1 = n
        elif i == 1:
            id2 = n
        else:
            break
    spe = Z(extended_id.id(id1, id2))

    expr1c = sp1*zchain
    expr2c = sp2*zchain
    exprce = zchain*spe
    if len(term.ops) > 3:
        expr1c = two_locality_hamiltonian(expr1c, extended_id)
        expr2c = two_locality_hamiltonian(expr2c, extended_id)
        exprce = two_locality_hamiltonian(exprce, extended_id)

    if term.coeff >= 0:
        terms = [7, zchain, -3*sp1, -3*sp2, spe*6, exprce*2, -1*expr1c, -1*expr2c, -4*sp1*spe, -4*sp2*spe, sp1*sp2]
        return addition_exprs(iter(terms), coeff=term.coeff)
    else:
        terms = [5, -1*zchain, -1*sp1, -1*sp2, spe*2, -2*exprce, expr1c, expr2c, -4*sp1*spe, -4*sp2*spe, 3*sp1*sp2]
        return addition_exprs(iter(terms), coeff=-term.coeff)

def two_locality_hamiltonian(expr, extended_id):
    if isinstance(expr, Term):
        return _locality_reduced_expr(expr, extended_id)
    expr = expr.simplify()
    sec_adder = SequentialAdderExpr()
    n_terms = len(expr.terms)
    for i, term in enumerate(expr.terms):
        sec_adder.add(_locality_reduced_expr(term, extended_id))
    return sec_adder.expr()

class LocalityForm():
    MAX_LOCALITY = 32
    def __init__(self):
        self.max_locality = 2
        self.extended_id = ExtendedId(self.MAX_LOCALITY)
        self.forms = [Z(0).to_expr(), (Z(0)*Z(1)).to_expr()]

    def _extend(self, locality):
        if locality <= self.max_locality:
            return
        l = self.max_locality
        self.max_locality = locality
        while l < locality:
            two_l_h = two_locality_hamiltonian(self.forms[-1]*Z(l), self.extended_id)
            self.forms.append(two_l_h)
            l += 1

    def get_form(self, n):
        return self.forms[n-1]

    def two_locality_from_term(self, term, extended_id):
        locality = len(term.ops)
        if locality <= 2:
            return term
        if locality > self.max_locality:
            if locality > self.MAX_LOCALITY:
                raise ValueError('Locality should less than 33.')
            self._extend(locality)

        ids = [n for n in term.n_iter()]
        def find_id(local_id):
            if local_id < self.MAX_LOCALITY:
                return ids[local_id]
            a, b = self.extended_id.src(local_id)
            return extended_id.id(find_id(a), find_id(b))

        form = self.get_form(locality)
        exprs = SequentialAdderExpr()
        for t in form:
            ops = []
            for n in t.n_iter():
                id = find_id(n)
                ops.append(Z(id))
            term = Term.from_ops_iter(iter(ops), t.coeff)
            exprs.add(term)
        return exprs.expr() * term.coeff

    def two_locality(self, expr, extended_id, verbose=False):
        if isinstance(expr, Term):
            return self.two_locality_from_term(expr, extended_id)
        exprs = SequentialAdderExpr()
        if verbose:
            print('  Reduce locality. N terms:{}'.format(len(expr.terms)))
        cnt = 0
        for term in expr:
            exprs.add(self.two_locality_from_term(term, extended_id))
            cnt += 1
            if verbose:
                progress = cnt/len(expr.terms)
                print('\r    {}{:.0f}%'.format('●'*int(progress*32), progress*100), end='')
        if verbose:
            print()
        return exprs.expr()

def ising_hamiltonian_matrix(hamiltonian, r_qubits, verbose=False):
    if not isinstance(hamiltonian, QubitOperator):
        raise NotImplementedError

    r = r_qubits
    def indice(i, j):
        return i*r + j
    def sigmaX_to_Z(i, j, k):
        if j == k:
            return Expr.zero()
        else:
            return (1.0 - Z(indice(i,j))*Z(indice(i,k))) / 2
    def sigmaY_to_Z(i, j, k):
        if j == k:
            return Expr.zero()
        else:
            return 1.0j*(Z(indice(i,j)) - Z(indice(i,k))) / 2
    def sigmaZ_to_Z(i, j, k):
        if j == k:
            return Z(indice(i, j))
        else:
            return (Z(indice(i,j)) + Z(indice(i,k))) / 2
    def I_to_Z(i, j, k):
        if j == k:
            return I
        else:
            return (1.0 + (Z(indice(i,j))*Z(indice(i,k)))) / 2
    ising_spin = {'X': sigmaX_to_Z,
                  'Y': sigmaY_to_Z,
                  'Z': sigmaZ_to_Z,
                  'I': I_to_Z}

    n_qubits = max_n(hamiltonian)+1

    # Construct base H'_{j=k}
    sec_adder = SequentialAdderExpr()
    n_terms = len(hamiltonian.terms.items())
    for qo_ops, coeff in hamiltonian.terms.items():
        ops = [ising_spin['I'](i, 0, 0) for i in range(n_qubits)]
        for n, c in qo_ops:
            ops[n] = ising_spin[c](n, 0, 0)
        sec_adder.add(multiply_exprs(ops, coeff))
    base_j = sec_adder.expr()
    if r == 1:
        return [[base_j]]

    # Construct base H'_{j!=k}
    sec_adder = SequentialAdderExpr()
    n_terms = len(hamiltonian.terms.items())
    cnt = 0
    for qo_ops, coeff in hamiltonian.terms.items():
        ops = [ising_spin['I'](i, 0, 1) for i in range(n_qubits)]
        for n, c in qo_ops:
            ops[n] = ising_spin[c](n, 0, 1)
        sec_adder.add(multiply_exprs(ops, coeff))
        cnt += 1
        if verbose:
            progress=cnt/n_terms
            print('\r  {}{:.0f}%'.format('●'*int(progress*32), progress*100), end='')
    if verbose:
        print()
    base_jk = sec_adder.expr()

    # Cast for each H'_{j, k} to base form
    H = []
    for j in range(r):
        H.append([])
        for k in range(r):
            if k < j:
                H[-1].append(Expr.zero())
                continue
            if j == k and j == 0:
                H[-1].append(base_j)
                continue
            if j == 0 and k == 1:
                H[-1].append(base_jk)
                continue

            terms = []
            if j == k:
                for term in base_j:
                    ops = (Z(n+j) for n in term.n_iter())
                    terms.append(Term.from_ops_iter(iter(ops), term.coeff))
            else:
                for term in base_jk:
                    ops = []
                    for n in term.n_iter():
                        if n % r == 0:
                            ops.append(Z(n+j))
                        else:
                            ops.append(Z(n-1+k))
                    terms.append(Term.from_ops_iter(iter(ops), term.coeff))
            H[-1].append(Expr.from_terms_iter(iter(terms)))
    return H

def matrix_from_ising_hamiltonian(hamiltonian):
    if not isinstance(hamiltonian, Expr):
        raise TypeError()

    hamiltonian = hamiltonian.simplify()
    qubits = hamiltonian.max_n()+1
    mat = np.zeros((qubits, qubits))
    for term in hamiltonian:
        if term.is_identity:
            continue
        if len(term.ops) == 1:
            i = tuple(term.n_iter())[0]
            mat[i, i] = term.coeff
        elif len(term.ops) == 2:
            i, j = tuple(term.n_iter())
            mat[i, j] = term.coeff
        else:
            raise ValueError("Locality of ising hamiltonian must be less-equal than 2.")

    return mat

def qubo_from_ising_hamiltonian(hamiltonian):
    if not isinstance(hamiltonian, Expr):
        raise TypeError()

    hamiltonian = hamiltonian.simplify()
    n_qubits = hamiltonian.max_n()+1

    mat = np.zeros((n_qubits, n_qubits))
    for term in hamiltonian:
        if term.is_identity:
            continue
        elif len(term.ops) == 1:
            i = tuple(term.n_iter())[0]
            mat[i, i] += -2.0*term.coeff
        elif len(term.ops) == 2:
            i, j = tuple(term.n_iter())
            mat[i, i] += -2.0*term.coeff
            mat[j, j] += -2.0*term.coeff
            mat[i, j] += 4.0*term.coeff
        else:
            print(term)
            raise ValueError("Locality of ising hamiltonian must be less-equal than 2.")

    return mat

def pauli_z_from_qubo_bit(qubo_bit, qubits=-1):
    if qubits >= 0:
        qubo_bit = [qubo_bit[i] if i < len(qubo_bit) else 0 for i in range(qubits)]
    return [1.0 - 2.0*q for q in qubo_bit]

def qubo_bit_from_pauli_z(pauli_z, qubits=-1):
    if qubits >= 0:
        pauli_z = [pauli_z[i] if i < len(pauli_z) else 1 for i in range(qubits)]
    return [(1.0-z)/2 for z in pauli_z]
