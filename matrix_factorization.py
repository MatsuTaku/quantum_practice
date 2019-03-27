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
    def __init__(self, V, k, steps=50, alpha=0.02, beta=0.02, threshold=0.001):
        self.V = V
        self.k = min(k, 1024)
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def run(self):
        n, m = len(self.V), len(self.V[0])
        W = np.random.rand(n, self.k)   # W is nonnegative matrix
        H = np.random.rand(self.k, m)       # H is binary matrix
        for step in range(self.steps):
            # Solve W
            for i in range(n):
                for j in range(m):
                    if self.V[i][j] == 0:
                        continue
                    error = (self.V[i][j] - np.dot(W[i], H[:, j])) ** 2
                    error += self.beta * ((la.norm(W[i]) ** 2))
                    for kk in range(self.k):
                        W[i][kk] += self.alpha * 2 * error * H[kk][j]
                        if W[i][kk] < 0:
                            W[i][kk] = 0

            # Solve H
            a = np.zeros((m, self.k))
            for i in range(m):
                for l in range(n):
                    vli = self.V[l][i]
                    if vli == 0:
                        continue
                    for j in range(self.k):
                        wlj = W[l][j]
                        a[i][j] += wlj * (wlj - vli*2)
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
            if error < self.threshold:
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
    # %%
    nP, nQ = MF_classic(R, int(len(R[0])*0.5)).run()
    print(nP.T)
    print(nQ)
    nR = np.dot(nP.T, nQ)
    print(nR)

    # %% qaoa
    nP, nQ = MF_quantam(R, int(len(R[0])*0.6)).run()
    print(nP)
    print(nQ)
    nR = np.dot(nP, nQ)
    print(nR)
