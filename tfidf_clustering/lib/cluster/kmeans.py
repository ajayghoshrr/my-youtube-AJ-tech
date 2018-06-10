"""
This module can able to do perform clustering based on the data given in the format of matrix.
***Use this link for reference: https://en.wikipedia.org/wiki/K-means_clustering ***
Module can perform k-means clustering on the rows of a matrix, given k and the matrix whose rows are \
to be clustered as input parameters
"""

__author__ = "Author: Ajaighosh Ramachandran"
__date__ = "Date: 2018-06-09 17:37:55 +0530 (Sat, 09 Jun 2018)"
__email__ = "ajayghoshrr@gmail.com"
__status__ = "Development"

# package imports
import numpy as np
import logging

logging.basicConfig(filename='KMeans.log', level=logging.INFO,
                    format='%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s')
class KMeans:
    """
    KMeans class will initiate the number of clusters, tolerance, and epochs

    ***INPUT PARAMETERS***
    Keyword arguments:
    n_clusters: The number of cluster wants to get formed. Default parameter value is 2, Default parameter.
    tolerance: minimum tolerance of average between centroids. difference of average less than tolerance, cluster\
     optimized, Default parameter
    epochs: number of iterations for finding best cluster, Default parameter
    """
    def __init__(self, n_clusters=2, tolerance=0.001, epochs=300):
        """
        KMeans Initialization
            :param n_clusters: No of clusters for KMeans
            :type: Int
            :param tolerance: Average tolerance for step and optimization
            :type: float
            :param epochs: No of Iteration
            :type: Int
        """
        self.k = n_clusters
        self.tol = tolerance
        self.max_iter = epochs
        logging.info("KMeans values initialized with n_clusters = {0}, tolerance = {1}, epochs = {2}".format(n_clusters,
                                                                                                             tolerance,
                                                                                                             epochs))

    # find the best fit centroids
    def fit(self, data):
        """
        Keyword arguments for fit
            :param data: It's an multidimensional matrix: Mandatory
            :type: Iterable
            :return: None. function will find the best fit for the clustering
        """
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
        logging.info("Initialized the centroids")
        for i in range(self.max_iter):
            logging.info("Started Iteration no : {0}".format(i))
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []
            for feature_set in data:
                # normalizing the centroid list
                distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature_set)
            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                logging.info("Finding the mean of all centroids")
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    logging.info("Optimizing.........")
                    optimized = False
            if optimized:
                break

    def predict(self, data):
        """
        Keyword parameters:
        :param data: It's an multidimensional matrix: Mandatory.
        :return: list
        """
        # normalizing the centroid list
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


