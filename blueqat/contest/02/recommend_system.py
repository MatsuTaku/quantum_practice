import numpy as np
from matrix_factorization import MF_quantum
import collaborative_filtering as cf
import time

class RecommendSystem:
    def __init__(self, R, qbits=4):
        max_id = np.max(np.max(R))
        V = np.zeros((len(R), max_id+1))
        for i in range(len(V)):
            for j in range(len(V[0])):
                V[i][j] = 0.01
        for i in range(len(R)):
            for id in R[i]:
                V[i][id] = 1

        self.W, self.H = MF_quantum(V, qbits).run()
        print(V)
        print(self.W)
        print(self.H)
        print(np.dot(self.W, self.H))

    def get_recommends(self, feature):
        feature_vec = np.zeros(len(self.H[0]), dtype=bool)
        for id in feature:
            feature_vec[id] = 1
        return cf.get_similarities(feature_vec, self.W, self.H)

if __name__ == '__main__':
    # %% sample
    bought_sets = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [1,2,5,6],
        [1,3,2,4,8,6]
    ])

    for qbits in range(1, 9):
        start = time.time()
        recsys = RecommendSystem(bought_sets, qbits)
        for set in bought_sets:
            print(recsys.get_recommends(set))
        print('mf time[', qbits, '] = ', time.time() - start)
