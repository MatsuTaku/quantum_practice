import numpy as np
from numpy import linalg as la
from blueqat.opt import Opt
import random
import time

class MF_classic:
    '''
    Matrix Factorization by classic computing
    '''
    def __init__(self, R, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
        self.R = R
        self.K = min(K, len(R[0]))
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def get_rating_error(self, r, p, q):
        return r - np.dot(p.T, q)

    def get_error(self, r, p, q, beta):
        error = 0.0
        for i in range(len(r)):
            for j in range(len(r[0])):
                if r[i][j] != 0:
                    error += self.get_rating_error(r[i][j], p[:, i], q[:, j]) ** 2
        error += beta * (la.norm(p)**2 + la.norm(q)**2)
        return error

    def run(self):
        print('Factorizing matrix with ', self.K, ' qubit...')
        n, m = len(self.R), len(self.R[0])
        P = np.random.rand(self.K, n)
        Q = np.random.rand(self.K, m)
        for step in range(self.steps):
            for i in range(n):
                for j in range(m):
                    if self.R[i][j] == 0:
                        continue
                    error = self.get_rating_error(self.R[i][j], P[:, i], Q[:, j])
                    for k in range(self.K):
                        delta_p = self.alpha * (2 * error * Q[k, j])
                        delta_q = self.alpha * (2 * error * P[k, i])
                        P[k, i] += delta_p
                        Q[k, j] += delta_q

            error = self.get_error(self.R, P, Q, self.beta)
            if error < self.threshold ** 2:
                break
        return P, Q


class MF_quantum:
    '''
    Preparing Nonnegative/Binary Matrix Factorization (NBMF) for input matrix.
    Note that account 0 to empty value of this system.
    '''

    def __init__(self, V, k, steps=5000, converge_steps=10, alpha=0.002, beta=0.02, threshold=0.001):
        self.V = V
        # self.k = min(k, len(V[0]))
        self.k = k
        self.steps = steps
        self.converge_steps = converge_steps
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def get_rating_error(self, v, w, h):
        return v - np.dot(w, h)

    def get_error(self, W, H):
        error = 0
        for i in range(len(W)):
            for j in range(len(H[0])):
                v = self.V[i][j]
                if v == 0:
                    continue
                error += self.get_rating_error(v, W[i], H[:, j])**2
        return error**0.5

    def optimize_W(self, W, H):
        '''
        Nonnegative Matrix Factorization (NMF) for W.
        '''
        n, m = len(W), len(H[0])
        for step in range(self.steps // self.converge_steps):
            for i in range(n):
                for j in range(m):
                    v = self.V[i][j]
                    if v == 0:
                        continue
                    error = self.get_rating_error(v, W[i], H[:, j])
                    for kk in range(self.k):
                        if H[kk][j]:
                            delta = self.alpha * (2 * error) # (* H[kk][j])
                            if W[i][kk] + delta >= 0:
                                W[i][kk] += delta
                            else:
                                W[i][kk] = 0 # value of W is nonnegative
            error = self.get_error(W, H)
            error += self.beta * la.norm(W)
            if error < self.threshold:
                break

    def optimize_H(self, W, H):
        '''
        Binary Matrix Factorization for H.
        '''
        n, m = len(W), len(H[0])
        # Construct parameter a
        a = np.zeros((m, self.k))
        for i in range(m):
            for l in range(n):
                v = self.V[l][i]
                if v == 0:
                    continue
                a[i] += np.multiply(W[l], W[l] - v*2)
        # Construct parameter b
        b = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(i+1, self.k):
                b[i][j] = np.dot(W[:, i], W[:, j])
        # Minimize hamiltonian
        for n in range(m):
            qubo = np.array(b)
            for i in range(self.k):
                qubo[i][i] = a[n][i]
            H[:, n] = Opt().add(qubo).run()

    def run(self):
        n, m = len(self.V), len(self.V[0])
        # Init W to nonnegative matrix
        W = np.random.rand(n, self.k)
        # Init H to binary matrix without zero column
        H = np.random.randint(0, 2, (self.k, m), dtype='bool')
        for i in range(m):
            if la.norm(H[:, i]) == 0:
                H[random.randint(0, self.k-1), i] = 1
        # Anneal W/H untill conveged
        prev_error = self.get_error(W, H)

        for converge_step in range(self.converge_steps):
            self.optimize_W(W, H)
            self.optimize_H(W, H)
            # Check conversion
            error = self.get_error(W, H)
            if prev_error > error and prev_error - error < self.threshold:
                break

        return W, H


if __name__ == '__main__':
    R = np.array([
        [5,3,0,1],
        [4,0,0,1],
        [1,1,0,5],
        [1,0,0,4],
        [0,1,5,4]
    ])
    print('R = \n', R)
    # %% classic MF
    print('--classic--')
    nP, nQ = MF_classic(R, 2).run()
    print('P = \n', nP.T)
    print('Q = \n', nQ)
    nR = np.dot(nP.T, nQ)
    print('P*Q = \n', nR)

    # %% Nonnegative/Binary matrix factorization by quantum approach
    print('--quantum--')
    nW, nH = MF_quantum(R, 2).run()
    print('W = \n', nW)
    print('H = \n', nH)
    nR = np.dot(nW, nH)
    print('W*H = \n', nR)
