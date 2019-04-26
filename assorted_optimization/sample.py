import numpy as np
from blueqat import opt, pauli
from blueqat.pauli import qubo_bit as q
import itertools

def expr_to_qubo(expr):
    assert type(expr) == pauli.Expr
    n = expr.max_n()+1
    qubo = np.zeros((n, n))
    for term in expr:
        if term.is_identity:
            continue
        if len(term.ops) == 1:
            i = tuple(term.n_iter())[0]
            qubo[i,i] += term.coeff
        if len(term.ops) == 2:
            i, j = tuple(term.n_iter())
            if i > j:
                i, j = j, i
            qubo[i,j] += term.coeff
    return qubo

def solver_minimum_assortment(A, L):
    N = len(A)
    M = len(A[0])
    L = N*M
    W = L + M - 1

    H1 = 0
    for i in range(N):
        h = -1
        for j in range(W):
            h += q(W*i+j)
        print(h)
        H1 += h ** 2
    print(expr_to_qubo(H1))

    area = []
    for i in range(N):
        l = 0
        while l < M and A[i][l] == 0:
            l+=1
        r = 0
        while r < M and A[i][M-1-r] == 0:
            r+=1
        area.append([M-1-l, W-M-1+r])

    H2 = 0
    for k in range(L):
        h = -0.5
        for i in range(N):
            for j in range(M):
                if A[i,j] == 0:
                    continue
                id = W*i+M-1+k-j
                if id < area[i][0] or id > area[i][1]:
                    continue
                h += q(id)
        H2 += h ** 2
    print(expr_to_qubo(H2))

    alpha = 1.0
    beta = 1.0
    H = H1*alpha + H2*beta

    qubo = expr_to_qubo(H)
    print(qubo)
    result = opt.Opt().add(qubo).run()
    print(result)

    panels = [0]*L
    for i in range(N):
        cnt = 0
        for j in range(W):
            cnt += result[i*N+j]
            if cnt > 1:
                print(i, j, cnt)
                return False
            if result[i*N+j] == 1:
                for k in range(j, j+M):
                    panels[k] += A[i,k-j]
                    if A[i,k-j] > 1:
                        return False
    return True



if __name__ == '__main__':
    A = np.array([[1,1,0,0],
                  [0,1,1,1],
                  [0,0,0,1]])

    l, r = 1, 12
    c = 0
    while l < r:
        c = (l+r)//2
        print('c', c)
        if solver_minimum_assortment(A, c):
            r = c-1
        else:
            l = c+1

    print(c)
