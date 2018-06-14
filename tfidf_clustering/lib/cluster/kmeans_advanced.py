"""
This module can able to do perform clustering based on the data given in the format of matrix.
***Use this link for reference: https://en.wikipedia.org/wiki/K-means_clustering ***
Module can perform k-means clustering on the rows of a matrix, given k and the matrix whose rows are \
to be clustered as input parameters
"""

__author__ = "Author: Ajaighosh Ramachandran"
__date__ = "Date: 2018-06-14 08:37:55 +0530 (Thu, 13 Jun 2018)"
__email__ = "ajayghoshrr@gmail.com"
__status__ = "Development"

# package imports
import numpy as np
import logging

logging.basicConfig(filename='KMeansAdvanced.log', level=logging.INFO,
                    format='%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s')

class KMeansAdvanced:
    """
    This module is intended to cluster the data based on the given number of clusters.
    """
    def __init__(self, n_clusters = 2, max_iter = 300, tol = 0.001):
        """
        Initializing the number of clusters, tolerance and maximum iteration
            :param n_clusters: No of clusters
            :type: Int
            :param max_iter: Number of Iteration
            :type: Int
            :param tol: Error tolerance
            :type: Float
        """
        self.K = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        logging.info("KMeans values initialized with n_clusters = {0}, max_iter = {1}, tol = {2}".format(n_clusters, max_iter, tol))


    def fit(self, dataset):
        """
        Function help to find the best cluster
            :param dataset: Multidimensional dataset to cluster
            :type: np array
            :return: Final centroid and classes
        """
        try:
            no_of_doc, no_of_features = dataset.shape
        except Exception as e:
            logging.debug("Error Raised due to Exception {0}. Please pass data type as np array".format(e))
            exit(1)
        # Initialized the K random centroid. Randomness sometimes helps to achieve good fit early.
        init_centroid = dataset[np.random.randint(0, no_of_doc-1, size=self.K)]
        # Initializing the previous centroid value as zero with same shape of init centroid.
        prev_centroid = np.zeros(init_centroid.shape)
        # Initializing the npArray of size no_of_doc and shape of 1 * no_of_doc
        # To store the final cluster or label details
        doc_store_init = np.zeros((no_of_doc, 1))
        # Finding the distance between the centroid
        centroid_distance = KMeansAdvanced.euclidian_distance(init_centroid, prev_centroid)
        # If the tolerance greater than centroid distance, the loop will run till convergence or it will run till max_iter
        while self.tol < centroid_distance and self.max_iter:
            logging.info("Optimizing iter remaining :: {0} and centroid_distance is ::{1}".format(self.max_iter,
                                                                                                  centroid_distance))
            centroid_distance = KMeansAdvanced.euclidian_distance(init_centroid, prev_centroid)
            prev_centroid = init_centroid.copy()
            logging.debug("Initialised centroid ::{0}".format(init_centroid))
            for index_dataset, val_dataset in enumerate(dataset):
                # Creating a multi dimensional list of value for k with 0
                logging.info("Iterating through the dataset index::{0}.".format(index_dataset))
                # To store distance to the val_dataset for each centroid
                distance_init = np.zeros((self.K,1))
                # Parsing through each centroid to find minimum distance
                for index_centroid, val_centroid in enumerate(init_centroid):
                    # Calculating the distance to each centroid and store value in Distance_init
                    distance_init[index_centroid] = KMeansAdvanced.euclidian_distance(val_centroid, val_dataset)
                # Find the smallest distance to centroid and store into doc_store_init - Note: argmin return INDEX of the array.
                # Distance contain K Distances
                # ARGMIN() will return the index of minimum distance value
                # Doc store contain the the Index value of minimum distance  ----- INDEX is cluster/class
                doc_store_init[index_dataset, 0] = np.argmin(distance_init)
            # Creating an empty space for storing new centroid after mean
            temp_centroids = np.zeros((self.K, no_of_features))
            for index in range(len(init_centroid)):
                # closest_instance = [i for i in range(len(doc_store_init)) if doc_store_init[i] == index]
                # finding the clusters having same centroid and finding the mean
                closest_instance = []
                for i in range(len(doc_store_init)):
                    if doc_store_init[i] == index:
                        closest_instance.append(i)
                init_centroid = np.mean(dataset[closest_instance], axis=0)
                # storing the init_centroid to the temp_centroids based on the index
                temp_centroids[index, :] = init_centroid.copy()
                logging.info("New centroids/init_centroid {0}".format(temp_centroids))
            init_centroid = temp_centroids.copy()
            self.max_iter -=1
        if self.max_iter == 0:
            logging.warning("Cluster optimization failed. Increase max_iter")
            raise Warning("Cluster optimization - Not converged - Increase max_iter")
        return init_centroid, doc_store_init





    @staticmethod
    def euclidian_distance(vec1, vec2):
        """

        :param vec1: Iterable
        :param vec2: Iterable
        :return: Euclidean distance
        :type: Decimal
        """
        return np.linalg.norm(vec1-vec2)

    @staticmethod
    def euclidian_distance_self(vec1, vec2):
        """

        :param vec1:
        :param vec2:
        :return:
        """
        import math
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(vec1, vec2)]))



# dataset = [[1,2,3], [4,5,6], [7,8,9],
#            [10, 11, 12], [13, 14, 15],
#            [16, 17, 18], [19, 20, 21],
#            [22, 23, 24], [25, 26, 27]]
# cl = KMeansAdvanced()
# print(cl.fit(np.array(dataset)))
