import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, data, numberOfCentroids):
        self.numberOfCentroids = numberOfCentroids
        self.centroids = np.random.rand(numberOfCentroids, len(data.transpose()))
        self.predictions = {}
        for i in range (0, numberOfCentroids):
            self.predictions[i] = []
        self.predictionsList = [] #To check for convergence
        self.previousPreddictionsList = [] #To check for convergence
        self.converged = False
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        pass
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        
        #while self.predictions[0] != self.predict(X)[0]:
        #pred, predList = self.predict(X)
        t = 0
        for i in range(100):
            t+=1
           # print("iteration: ", t)
            self.predictions = self.predict(X)
            checkpoint = self.centroids.copy()
            for i in range(len(self.centroids)):
                for m in range(len(np.transpose(X))):
                    if self.predictions[i]: #If list is empty, calculation in next line will crash.
                        self.centroids[i][m] = np.mean(np.transpose(self.predictions[i])[m])

            self.centroids = np.flip(self.centroids, 0)            
            #print("prev: ", checkpoint)
            #print("curr: ", self.centroids)
            #print("PLS: ", checkpoint == self.centroids)
        self.converged = True
            #TODO: update centroid points, should be finished w/first dataset after this.

        
        #return self.centroids
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        #predictions = self.predictions.copy()
        predictions = {}
        for i in range (0, self.numberOfCentroids):
            predictions[i] = []
        predList = []
        for elem in X:
            key = np.argmax(euclidean_distance(self.centroids, elem))
            predList.append(key)
            predictions[key].append(elem)
        if self.converged:
            return predList
        
        print(self.predictionsList == predList)
        #print(predictions == self.predictions)
        self.predictionsList = predList
        #print(self.predictionsList == predList)
        return predictions
    
    def euclideanDistance(self, centroid, x):
        return np.linalg.norm(centroid, x)
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return np.array(self.centroids)
    
    
    
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
   
    X, z = np.asarray(X), np.asarray(z)
    print("X: ", X.shape)
    print("Z: ", z.shape)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += sum(((Xc - mu) ** 2).sum(axis=1))
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  