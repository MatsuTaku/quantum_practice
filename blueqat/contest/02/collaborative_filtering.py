import numpy as np
import numpy.linalg as la

def cosign_similarity(A, B):
    return np.dot(A, B) / (la.norm(A) *  la.norm(B))

def get_similarities(F, W, H):
    sims = [cosign_similarity(F, np.dot(W[i], H)) for i in range(len(W))]
    sorted_sims = np.sort(sims)
    sorted_ids = np.argsort(sims)
    return [[id, sim] for (id, sim) in zip(sorted_ids[::-1], sorted_sims[::-1])]
