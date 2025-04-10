#A
#librairies
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
#distance matrix
distanceMatrix = np.array([[5,0,0,0,0,0,0], [8,6,0,0,0,0,0], [4,4,5,0,0,0,0], [7,5,1,4,0,0,0], [7,4,2,4,1,0, 0], [8,3,7,7,7,5,0], [2,4,6,1,5,5,8]])

#heirarchal clustering using single linkage
clustering = linkage(distanceMatrix, 'single')

#dendogram
plt.figure(figsize=(10, 5))
dendrogram(clustering, labels=[f'Point {i+1}' for i in range(len(distanceMatrix))])
plt.title('Hierarchical Clustering Dendrogram Problem 4a')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()


#b
from sklearn.neighbors import NearestNeighbors

def isCorePoint(distanceMatrix, pointIndex, epsilon, min):
     point = distanceMatrix[pointIndex]
     distances = np.linalg.norm(distanceMatrix - point, axis=1)
     numNeighbors = np.sum(distances <= epsilon)
     return numNeighbors >= min

epsilon = 6
min = 2
pointIndex = 0

isCore = isCorePoint(distanceMatrix, pointIndex, epsilon, min)
print(isCore)