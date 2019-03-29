import numpy as np
from numpy import linalg as la
from blueqat import vqe, opt
from blueqat.pauli import qubo_bit as q
import random

class MF_classic:
    def __init__(self, R, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
        self.R = R
        self.K = K
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
                error += self.get_rating_error(r[i][j], p[:, i], q[:, j]) ** 2
                error += beta * (np.linalg.norm(p[:, i])**2 + np.linalg.norm(q[:, j])**2)
        return error

    def run(self):
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


class MF_quantam:
    def __init__(self, V, k, steps=5000, alpha=0.002, beta=0.02, threshold=0.001):
        self.V = V
        self.k = min(k, 1024)
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def run(self):
        n, m = len(self.V), len(self.V[0])
        # Init W to nonnegative matrix
        W = np.random.rand(n, self.k)
        # Init H to binary matrix without zero column
        H = np.random.randint(0, 2, (self.k, m), dtype='bool')
        # Anneal W/H untill conveged
        prev_error = la.norm(self.V - np.dot(W, H))
        for upper_step in range(10):
            # Solve W
            for step in range(self.steps // 10):
                for i in range(n):
                    for j in range(m):
                        v = self.V[i][j]
                        if v == 0:
                            continue
                        error = v - np.dot(W[i], H[:, j])
                        for kk in range(self.k):
                            if H[kk][j]:
                                W[i][kk] += self.alpha * (2 * error)
                error = la.norm(self.V - np.dot(W, H))
                error += self.beta * la.norm(W)
                if error < self.threshold:
                    break

            # Solve H
            a = np.zeros((m, self.k))
            for i in range(m):
                for l in range(n):
                    v = self.V[l][i]
                    if v == 0:
                        continue
                    for j in range(self.k):
                        w = W[l][j]
                        a[i][j] += w * (w - v*2)
            b = np.zeros((self.k, self.k))
            for i in range(self.k):
                for j in range(i+1, self.k):
                    b[i][j] = np.dot(W[:, i], W[:, j])
            for n in range(m):
                hamiltonian = 0
                for i in range(self.k):
                    hamiltonian += q(i) * a[n][i]
                    for j in range(i+1, self.k):
                        hamiltonian += q(i)*q(j)*b[i][j]

                result = vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, 2)).run()
                H[:, j] = result.most_common(1)[0][0]

            error = la.norm(self.V - np.dot(W, H))
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
    # %% classic MF
    nP, nQ = MF_classic(R, 2).run()
    print(nP.T)
    print(nQ)
    nR = np.dot(nP.T, nQ)
    print(nR)

    # %% Nonnegative/Binary matrix factorization by quantum approach
    nP, nQ = MF_quantam(R, 2).run()
    print(nP)
    print(nQ)
    nR = np.dot(nP, nQ)
    print(nR)
