import numpy as np
import numpy.linalg as la

def cosine_similarity(A, B):
    size = la.norm(A) *  la.norm(B)
    if size != 0:
        return np.dot(A, B) / size
    else:
        return 0.0

def jaccard_similarity_coefficient(A, B):
    maxs = [max(float(a), float(b)) for a, b in zip(A, B)]
    maxs = np.array(maxs)
    size = la.norm(maxs, ord=1)
    if size != 0:
        return np.dot(A, B) / size
    else:
        return 0.0

def dice_similarity_coefficient(A, B):
    size = la.norm(A, ord=1) + la.norm(B, ord=1)
    if size != 0:
        return 2*np.dot(A, B) / size
    else:
        return 0.0

def simpson_similarity_coefficient(A, B):
    size = min(la.norm(A, ord=1), la.norm(B, ord=1))
    if size != 0:
        return np.dot(A, B) / size
    else:
        return 0.0

def custom_similarity_coefficient(A, B):
    '''
    Hybrid similarity from large ratio of 'simpson' and small ratio of 'jaccard'.
    '''
    p = 0.8
    simpson = simpson_similarity_coefficient(A, B)
    jaccard = jaccard_similarity_coefficient(A, B)
    return simpson*p + jaccard*(1.0-p)
