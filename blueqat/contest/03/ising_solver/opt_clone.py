from blueqat.opt import *
from blueqat.pauli import Expr
from .transform import matrix_from_ising_hamiltonian
from .utils import get_energy, get_coeffs
import numpy as np

class OptClone(Opt):
    def add_ising(self, hamiltonian):
        self.qubo = []
        if isinstance(hamiltonian, Expr):
            hamiltonian = matrix_from_ising_hamiltonian(hamiltonian)
        self.J = np.array(hamiltonian)
        return self

    def run(self,shots=1,targetT=0.02,verbose=False):
        """
        Run SA with provided QUBO.
        Set qubo attribute in advance of calling this method.

        Return value form as ising spin grass.
        """
        if self.qubo != []:
            self.qi()
        J = self.reJ()
        N = len(J)

        '''itetemp
        Original is const value(100).
        '''
        itetemp = N*2
        '''Rtemp
        Original is 0.75
        '''
        Rtemp = 0.90

        self.E = []
        qq = []
        for i in range(shots):
            T = self.Ts
            q = np.random.choice([-1,1],N)
            EE = []
            EE.append(Ei(q,self.J)+self.ep)
            while T>targetT:
                x_list = np.random.randint(0, N, itetemp)
                for x in x_list:
                    q2 = np.ones(N)*q[x]
                    q2[x] = 1
                    dE = -2*sum(q*q2*J[:,x])

                    if dE < 0 or np.exp(-dE/T) > np.random.random_sample():
                        q[x] *= -1
                EE.append(Ei(q,self.J)+self.ep)
                T *= Rtemp
            self.E.append(EE)
            # qtemp = (np.asarray(q,int)+1)/2
            # qq.append([int(s) for s in qtemp])
            '''
            Return ising spin as it is
            '''
            ztemp = np.asarray(q,int)
            qq.append(ztemp)
            if verbose == True:
                # print(i,':',[int(s) for s in qtemp])
                print(i,':',ztemp)
            if shots == 1:
                qq = qq[0]
        if shots == 1:
            self.E = self.E[0]
        return qq


class OptAnealer:
    def __init__(self):
        self.H = Expr.zero()
        self.eigenvalue = .0
        self.eigenstate = []
        self.Ts = 100
        self.Tf = 0.02

    def hi(self):
        self.H = self.H.simplify()

        N = self.H.max_n()+1
        M = len(self.H.terms)
        self.hb = np.zeros((M, N), dtype=bool)
        for i, term in enumerate(self.H):
            for n in term.n_iter():
                self.hb[i, n] = 1

    def add(self, hamiltonian):
        if not isinstance(hamiltonian, Expr):
            raise TypeError()
        self.H = hamiltonian
        return self

    def run(self, shots=1, verbose=False, itetemp=300, Rtemp=0.70):
        '''
        Run sa with provided ising hamiltonian.
        Set hamiltonian in advance of calling this method.
        '''
        self.hi()

        N = self.H.max_n()+1

        self.E = []
        ss = []
        if verbose:
            print('shots:{} N:{}, terms:{}'.format(shots, N, len(self.H.terms)))
        for i in range(shots):
            T = self.Ts
            targetT = self.Tf
            s = np.random.choice((-1,1),N)
            hc = np.array(get_coeffs(self.H, s))
            EE = []
            EE.append(sum(hc))
            while T > targetT:
                x_list = np.random.randint(0, N, itetemp)
                dE = .0
                for x in x_list:
                    coeffs = hc*self.hb[:, x]
                    dE = -2*sum(coeffs) * s[x]
                    if dE < 0 or np.exp(-dE/T) > np.random.random_sample():
                        s[x] *= -1
                        hc -= coeffs * 2
                EE.append(sum(hc))
                T *= Rtemp
            self.E.append(EE)
            ss.append((EE[-1], s))
            if verbose:
                print(i, ':', ss[i])
            if shots == 1:
                ss = ss[0]
        if shots == 1:
            self.E = self.E[0]
        return ss
