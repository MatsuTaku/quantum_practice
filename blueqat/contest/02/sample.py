import numpy as np
from recommend_system import RecommendSystem
import time

if __name__ == '__main__':
    '''
    Sample table
    xaxis: User
    yaxis: item
        value: evaluation
    '''
    evaluation_table = np.array([ # evaluation 1 - 5. 0 is empty.
        [1,2,3,4,5,0,0,0],
        [0,0,0,1,2,3,4,5],
        [5,4,3,2,0,0,1,0],
        [1,0,2,3,0,5,0,4]
    ])
    print(evaluation_table)

    results = []
    for qbits in range(4, 9):
        start = time.time()
        recsys = RecommendSystem(evaluation_table, qbits)
        print('Result: (qbits=', qbits)
        print('Recommend items (item(index), rate)')
        for id, set in enumerate(evaluation_table):
            recommends = recsys.get_recommends(set)
            print(id, '<', recommends)
        t = time.time() - start
        results.append([qbits, t, recsys.error])

    print('qbits\t| time(s)\t| error(|V-WH|)')
    for qbits, time, error in results:
        print(qbits, '\t| ', int(time), '\t| ', error)
