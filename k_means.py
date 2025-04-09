import pandas as pd
import numpy as np

Point=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
data = {
    'X': [2, 2, 11, 6, 6, 1, 5, 4, 10, 7, 9, 4, 3, 3, 6],
    'Y': [10, 6, 11, 9, 4, 2, 10, 9, 12, 5, 11, 6, 10, 8, 11]
}

df = pd.DataFrame(data,index=[_ for _ in Point])

'''
Centroid 1=(2,6) is associated with cluster 1.
Centroid 2=(5,10) is associated with cluster 2.
Centroid 3=(6,11) is associated with cluster 3.
'''
central=['C1','C2','C3']
centroids={
    'X':[2,5,6],
    'Y':[6,10,11]
}
central_cee=pd.DataFrame(centroids,index=[_ for _ in central])

def eucl_dist(centroid,point):
    return np.sqrt(((point['X']-centroid['X'])**2)+((point['Y']-centroid['Y'])**2))

def assignments(data,centroids,k=3):  
    assignments=[]
    for j in range(data.shape[0]):
        distances=[]
        for i in range(k):
            d=eucl_dist(centroids.iloc[i],data.iloc[j])
            distances.append(d)
            optimal_distance_index=np.argmin(distances)
            assignments.append((data.index[j],centroids.index[optimal_distance_index]))
    return assignments

def updating_centroids(assignments,curr_centroids,data,k=3):
    # Organize results by cluster
    clusters = {
    'C1': [],
    'C2': [],
    'C3': []
        }
    for centroid_idx in curr_centroids.index:
        # Get points assigned to this cluster
        cluster_points = [point for point, cluster in assignments if cluster == centroid_idx]
        
    # Create a new DataFrame for updated centroids
    new_centroids = curr_centroids.copy()
    
    # Update each centroid based on mean of points in the cluster
    for cluster_key, point_indices in clusters.items():
        if point_indices:  # Check if there are points in this cluster
            # Get X and Y values for points in this cluster
            x_values = [data.loc[point]['X'] for point in cluster_points]
            y_values = [data.loc[point]['Y'] for point in cluster_points]
            
            # Calculate mean positions
            x_mean = sum(x_values) / len(x_values)
            y_mean = sum(y_values) / len(y_values)
            
            # Update centroid
            new_centroids.loc[cluster_key, 'X'] = x_mean
            new_centroids.loc[cluster_key, 'Y'] = y_mean
    return new_centroids, clusters
        



def k_means(df,centroids,k=3):
    assigned=assignments(df,centroids)
    return updating_centroids(assigned,df,centroids)
       

        


        
        
        
        




print(df)
print(central_cee)
print(df.shape[0])
print(np.argmin(central_cee.iloc(0)))
print(central_cee.index[0])
up,clust=k_means(df,centroids=central_cee)
print(up)
print(clust)