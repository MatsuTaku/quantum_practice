from .transform import *
from .utils import max_n, get_energy
from .exprtools import *
from .opt_clone import OptClone, OptAnealer
from blueqat.pauli import *
from blueqat.opt import Opt
from openfermion.ops import QubitOperator
import numpy as np
import numpy.linalg as la
import itertools

class ElectronicStructureCalculator:
    def __init__(self, hamiltonian, r_qubits, verbose=False):
        if not isinstance(hamiltonian, QubitOperator):
            raise NotImplementedError

        self.n_qubits = max_n(hamiltonian)+1
        self.r_qubits = r_qubits
        self.verbose = verbose
        self.log(' Construct ising hamiltonian. r = {}'.format(r_qubits))
        self.hamiltonians = ising_hamiltonian_matrix(hamiltonian, r_qubits, verbose=verbose)

    def log(self, text='', end='\n'):
        if not self.verbose:
            return
        print(text, end=end)

    def get_hamiltonian(self, j, k):
        if j > k:
            j, k = k, j
        return self.hamiltonians[j][k]

    def H(self, sign):
        h = SequentialAdderExpr()
        for j, k in itertools.product(range(self.r_qubits), range(self.r_qubits)):
            h.add(self.get_hamiltonian(j, k) * sign[j]*sign[k])
        return h.expr()

    def C(self, sign):
        '''
        This method take time complexity as O(2^n*n*r^2).
        Actual algorithm is equivalent to follow:
            exprs = SequentialAdderExpr()
            for pat in range(2**n):
                ts = SequentialAdderExpr()
                for i in range(r):
                    tts = SequentialMultiplierExpr()
                    for j in range(n):
                        if (pat >> j) & 1:
                            tts.mul((1-Z(j*r+i))/2)
                        else:
                            tts.mul((1+Z(j*r+i))/2)
                    ts.add(tts.expr()*sign[i])
                powed = efficient_pow2_expr(ts.expr())
                exprs.add(powed)
            return exprs.expr()
        that is hard to execute.
        '''
        r = self.r_qubits
        n = self.n_qubits

        adder = SequentialAdderExpr()
        adder.add(Expr.from_number(r))
        coeff = 1.0/(2**n)*2
        for npat in range(2**n):
            for j, k in itertools.combinations(range(r), 2):
                ops = []
                for i in range(n):
                    if (npat >> i) & 1:
                        ops += [Z(r*i+j), Z(r*i+k)]
                s = sign[j] * sign[k]
                term = Term.from_ops_iter(iter(ops), coeff*s)
                adder.add(term)
        return adder.expr()

    def get_ground_state_energy(self, shots=10, threshold=-0.000001):
        min_eigenvalues = []
        rn_qubits = self.n_qubits * self.r_qubits
        extended_id = ExtendedId(rn_qubits)
        locality_form = LocalityForm()
        for i in range(-1, self.r_qubits//2):
            y = 10.0
            sign = [-1 if k == i else 1 for k in range(self.r_qubits)]
            self.log('   S\'{}'.format(sign))
            H = self.H(sign)
            C = self.C(sign)
            ## 2-locality calculation (rejected)
            # hi = locality_form.two_locality(h, extended_id, verbose=self.verbose)
            # ci = locality_form.two_locality(c, extended_id, verbose=self.verbose)
            # self.log('   locality: {}'.format(locality_form.max_locality))
            # self.log('   anealing qubits: {}'.format(hi.max_n()+1))
            errors = 0
            while True: # Monotonic decreasing for y
                # results = OptClone().add_ising(hi - ci*y).run(shots=shots, verbose=False)
                results = OptAnealer().add(H - C*y).run(shots=shots, verbose=False)
                if shots == 1:
                    results = [results]
                eig = 999999
                cb = 0
                # Getting minimum eigenvalue and eigenstate
                for e, s in results:
                    ccb = get_energy(C, s)
                    if ccb == 0:
                        continue
                    # print('|', s, '>', e, 'b', ccb)
                    if e < eig:
                        cb = ccb
                        eig = e
                self.log('     eigenvalue: {:.4f}, λ: {:.4f}, b: {}'.format(eig, y, cb))
                if cb == 0:
                    errors += 1
                    y = 10.0 ** errors+1
                    if errors >= 5:
                        break
                    continue
                if eig >= threshold:
                    break
                y += eig / cb

            min_eigenvalues.append(y)

        # print('λ...', min_eigenvalues)
        return min(min_eigenvalues)
