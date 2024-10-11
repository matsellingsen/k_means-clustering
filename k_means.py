import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class KMeans:
    
    def __init__(self, data, numberOfCentroids=2):
        self.numberOfCentroids = numberOfCentroids
      #  self.data = self.normalizeData(data)
        self.centroids = self.initClusters(data, numberOfCentroids)
        self.predictions = {}
        for i in range (0, numberOfCentroids):
            self.predictions[i] = []
        self.predictionsList = [] #To check for convergence
        self.converged = False
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        pass

    def normalizeData(self, data):
        min_val = np.min(data)
        max_val = np.max(data)

        scaled_matrix = (data - min_val) / (max_val - min_val)
        return scaled_matrix


    def initClusters(self, data, numberOfCentroids): #kMeans++ method (smart cluster initialization)
        clusters = []

        #Finding relevant dimensions for problem-space. (assuming 2D problems-space)
        maxX = np.max(data.transpose()[0]) 
        maxY = np.max(data.transpose()[1])

        c = np.multiply(np.array([maxX, maxY]), np.random.rand(1, len(data.transpose()))) #Initialize first cluster randomly
        clusters.append(c[0])
        for i in range(numberOfCentroids-1): 
            distances = euclidean_distance(data, c)
            probabilities = self.normalize(distances) #Calculating prob for each element to be next centroid.
            nextPoint = np.random.choice(distances, p=probabilities) #Picking next centroid.
            nextPoint = np.where(distances==nextPoint)[0][0]
            c = data[nextPoint]
            clusters.append([c[0], c[1]])
            
        return np.array(clusters)
    
    def normalize(self, x):
        newX = []
        for i in range(len(x)):
            newX.append(x[i] / sum(x))
        return newX
    
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
       # X = self.normalizeData(X)
        #while self.predictions[0] != self.predict(X)[0]:
        #pred, predList = self.predict(X)
        t = 0
        while not self.converged:
            t+=1
            #print("iteration: ", t)
            self.predictions, predList = self.predict(X)
            for i in range(len(self.centroids)):
                for m in range(len(np.transpose(X))):
                    if self.predictions[i]: #If list is empty, calculation in next line will crash.
                        #print(f"pred{i}: ", self.predictions[i])
                        self.centroids[i][m] = np.mean(np.transpose(self.predictions[i])[m])  

            if predList == self.predictionsList: #Check for convergence, i.e. that cluster members have not changed between two iterations.
                print(f"converged at iteration {t}")
                self.converged = True
            else: self.predictionsList = predList
    
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
       # X = self.normalizeData(X)
        predictions = {}
        for i in range (0, self.numberOfCentroids):
            predictions[i] = []
        predList = []
        for elem in X:
            key = np.argmin(euclidean_distance(self.centroids, elem))
            predList.append(key)
            predictions[key].append(elem)

        if self.converged:
            return np.array(predList)
        return predictions, predList
  
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
        return self.centroids
    
    
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

def manhattan_distance(x, y):
    distances = []
    for c in x:
        distances.append(np.sum(np.abs(c-y)))
    return np.array(distances)

def cosine_similarity(x, y):
    cos_sim = []
    for c in x:
        cos_sim.append(np.dot(c, y)/(np.linalg.norm(c)*np.linalg.norm(y)))
    return cos_sim

def cross_euclidean_distance(x, y=None):

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
  