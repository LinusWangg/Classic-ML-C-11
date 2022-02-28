from sklearn.cluster import KMeans
import numpy as np

data = np.random.randn(1000,5)
data2 = np.random.randn(1, 5)
kmeans = KMeans( 
                n_clusters = 3,
                n_init = 10,
                max_iter = 300,
                init = 'k-means++',
                ).fit(data)
center_id = kmeans.predict(data2)
print(kmeans.cluster_centers_[0])
print(np.linalg.norm(data2-kmeans.cluster_centers_[center_id]))

l = kmeans.cluster_centers_.tolist()
print(np.linalg.norm(data2-l[center_id.item()]))
print(l)