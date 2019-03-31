import numpy as np
from matrix_factorization import MF_quantum
import collaborative_filtering as cf
import time

class RecommendSystem:
    '''
    Note that let 0 as empty value of this system.
    '''
    EMPTY = 0.0

    def __init__(self, R, qbits):
        mf = MF_quantum(R, qbits)
        self.W, self.H = mf.run()
        self.error = mf.get_error(self.W, self.H)
        # print('    ')
        # print(V)
        # print(self.W)
        # print(self.H)
        # print(np.dot(self.W, self.H))

    def V_user(self, i):
        return np.dot(self.W[i], self.H)

    def V_item(self, i):
        return np.dot(self.W, self.H[:, i].T)

    def get_similarities(self, F):
        num_users = len(self.W)
        sims = [cf.cosine_similarity(F, self.V_user(i)) for i in range(num_users)]
        return np.array(sims)

    def get_recommends(self, evaluations, num=0):
        '''
        Must recommend item only empty evaluation in input evaluations.
        '''
        similarities = self.get_similarities(evaluations)
        recommendations = [np.dot(self.V_item(i), similarities) / len(self.W) for i in range(len(evaluations))]
        sorted_ids = np.argsort(recommendations)
        rec_items = []
        count = 0
        if num == 0:
            num = len(evaluations)
        for id in sorted_ids[::-1]:
            if count >= num:
                break
            if evaluations[id] != self.EMPTY:
                continue
            rec_items.append((id, recommendations[id]))
            count += 1
        return rec_items
