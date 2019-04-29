import numpy as np
from blueqat import opt
from blueqat.pauli import qubo_bit as q
import itertools
from opt import ConstOpt

def make_pieces(N, M):
    A = np.zeros((N, M), dtype=bool)
    for i in range(N):
        n = np.random.randint(1, M)
        for j in range(n):
            A[i,np.random.randint(M)] = 1
    return A


def solver_minimum_assortment(A, L, verbose=False):
    N = len(A)
    M = len(A[0])
    if L < M:
        return False
    W = L - M + 1
    Q = N * W
    # H1 = opt.zeros(Q)
    # for i in range(N):
    #     for j in range(W):
    #         for k in range(j,W):
    #             H1[W*i+j,W*i+k] = -1 if j == k else 2

    H2 = opt.zeros(Q)
    for a in range(N):
        for b in range(a+1, N):
            for i in range(-M+1, M):
                if M+abs(i) > L:
                    continue
                cnt = 0
                for k in range(max(0,i), min(0,i)+M):
                    if A[a,k] == 1 and A[b,k-i] == 1:
                        cnt += 1
                for k in range(W-abs(i)):
                    x = W*a+k + (0 if i >= k else -i)
                    y = W*b+k + (i if i >= k else 0)
                    H2[x, y] = cnt

    # print(H1)
    # print(H2)

    alpha = 1.0
    beta = 1.0
    # qubo = H1*alpha + H2*beta
    qubo = H2
    # print(qubo)
    shots = 100
    for _ in range(shots):
        result = ConstOpt().add(qubo).const_col_run(W)
        O = np.array(result, dtype=bool).reshape((N, W))
        panels = np.zeros((N, L), dtype=int)
        errors = 0
        for i in range(N):
            cnt = 0
            for j in range(W):
                cnt += O[i,j]
                if O[i,j] == 0:
                    continue
                for k in range(M):
                    panels[i, k+j] += A[i,k]
            errors += abs(cnt-1)
        field = np.zeros(L)
        for j in range(L):
            cnt = 0
            for i in range(N):
                cnt += panels[i,j]
            field[j] = cnt
            if cnt > 1:
                errors += cnt-1
        if verbose:
            print(np.asarray(O,int))
            print(field)
            print(panels)
        if errors == 0:
            return True
    return False


def quantum_assortment(A):
    sum = 0
    for l in A:
        for p in l:
            sum += p
    l, r = int(sum), N*M
    res = r
    while l <= r:
        c = (l+r)//2
        print('c', c)
        if solver_minimum_assortment(A, c):
            res = c
            r = c-1
        else:
            l = c+1

    return res


def sequential_assortment(A):
    field = []
    M = len(A[0])
    for p in A:
        s = 0
        insertable = False
        while not insertable:
            insertable = True
            if len(field) < s+M:
                field += [0]*(s+M-len(field))
            insertable = True
            for i in range(len(p)):
                if p[i] == 0:
                    continue
                if field[s+i] == 1:
                    insertable = False
                    break
            s+=1
        s-=1
        for i in range(M):
            field[s+i] += p[i]

    return len(field)


if __name__ == '__main__':
    # A = np.array([[0,1,0,0],
    #               [0,1,1,0],
    #               [1,0,1,0],
    #               [0,1,1,1],
    #               [1,1,0,1],
    #               [0,1,0,1],
    #               [0,0,1,1],
    #               [1,0,0,1]])

    N = 0xf
    M = 8
    A = make_pieces(N, M)
    print(np.asarray(A, int))

    print('quantum L', quantum_assortment(A))
    print('sequential L', sequential_assortment(A))
