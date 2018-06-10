"""

    This module contains rhe test case for the KMeans algorithms

"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from lib.cluster import kmeans

class TestKmeans:
    main_obj = kmeans.KMeans()
    def test_kMeans_init_test(self):
        print(kmeans.KMeans.__doc__)
        obj = kmeans.KMeans(n_clusters=4, tolerance=0.01, epochs= 500)
        if obj.k == 4 and obj.max_iter == 500 and obj.tol == 0.01:
            assert True
    def test_Kmeans_fit_test(self):
        print(kmeans.KMeans.fit.__doc__)
        obj = kmeans.KMeans()
        from sklearn import preprocessing
        data = [[1,2,3],
                [2,3,4],
                [9,8,10],
                [1,5,7]]
        data = preprocessing.scale(data)
        obj.fit(data)
        assert (obj.predict([0,0,0])) == 1