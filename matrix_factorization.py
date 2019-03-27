import numpy as np
from blueqat import vqe, opt
from blueqat.pauli import *

def get_rating_error(r, p, q):
    return r - np.dot(p.T, q)

def get_error(r, p, q, beta):
    error = 0.0
    for i in range(len(r)):
        for j in range(len(r[0])):
            error += get_rating_error(r[i][j], p[:, i], q[:, j]) ** 2
            error += beta * (np.linalg.norm(p[:, i])**2 + np.linalg.norm(q[:, j])**2)
    return error

def matrix_factorization_classic(R, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
    P = np.random.rand(K, len(R))
    Q = np.random.rand(K, len(R[0]))
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[0])):
                if R[i][j] == 0:
                    continue
                error = get_rating_error(R[i][j], P[:, i], Q[:, j])
                for k in range(K):
                    delta_p = alpha * (2 * error * Q[k, j])
                    delta_q = alpha * (2 * error * P[k, i])
                    P[k, i] += delta_p
                    Q[k, j] += delta_q

        if get_error(R, P, Q, beta) < threshold:
            break
    return P, Q

def matrix_factorization(R, K, steps=4, alpha=0.0002, beta=0.02):
    W = np.random.rand(K, len(R))
    H = np.random.rand(K, len(R[0])) % 2


if __name__ == '__main__':
    R = np.array([
        [5,3,0,1],
        [4,0,0,1],
        [1,1,0,5],
        [1,0,0,4],
        [0,1,5,4]
    ])
    # %%
    nP, nQ = matrix_factorization_classic(R, 2)
    nR = np.dot(nP.T, nQ)
    print(nR)

    # %% quantm
    nP, nQ = matrix_factorization(R, 2)
    nR = np.dot(nP.T, nQ)
    print(nR)
