##############################################################################################################
                                        # Anomaly Detection
##############################################################################################################

# In the attached code we will use the Iris dataset, we will relabel two Setosa entries as Versicolor
# and check to see if the two Unsupervised learning models (K means clustering and Density based clustering method
# (DBSCAN)) will detect the mis-labelled entries. Please note we would never assume that
# any cases which are flagged from these models are anomalies without consulting with an expert in the field.

# Import packages
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.cluster import DBSCAN

# Import the iris dataset
iris = sns.load_dataset("iris")

# Typical queries used to evaluate your data - always carry this out before completing any analysis
# on your data
iris.head() # the first 4 flowers are setosa
iris.info()
iris.describe()
iris.columns
iris.isnull().sum() # there are no null values in the data

# What are the unique flower types?
# setosa, versicolor and virginica.
iris['species'].unique()

# Rename the first and fourth species as Versicolor
# We will check to see if the unsupervised models will flag these cases as anomalies
iris.iloc[0,4] = 'versicolor'
iris.iloc[3,4] = 'versicolor'
iris.head() # the first and fourth species have been changed to Versicolor

# Only want to look at the Versicolor flowers
versi_df =  iris.loc[iris['species']=='versicolor']
versi_df = versi_df.reset_index(drop=True)
versi_df.head()

# Scale the data using the MinMaxScaler
# You should always scale your data when you are looking at distance based models
versi_np = versi_df.iloc[:,:4].values
scaler = MinMaxScaler()
versi_sc = scaler.fit_transform(versi_np)



# Visualising the data to see if the mis-labelled cases look like anomalies
# plt.clf()
sns.set()
plt.subplot(1,2,1)
sns.scatterplot(data=versi_df, x='sepal_length', y='sepal_width')
plt.scatter(x=5.1, y=3.5, marker='X')
plt.scatter(x=4.6, y=3.1, marker='X')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.title('Sepal length v width')

plt.subplot(1,2,2)
sns.scatterplot(data=versi_df, x='petal_length', y='petal_width')
plt.scatter(x=1.4, y=0.2, marker='X')
plt.scatter(x=1.5, y=0.2, marker='X')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Petal length v width')
plt.show()


##############################################################################################################
                                        # K means
##############################################################################################################


"""# Using the elbow method in K means clustering to find the optimal number of clusters 
   # WCSS is the sum of the squared distances from each point in a cluster to the centre of the cluster.
   # init refers to the initial cluster centres. k-means ++ speeds up convergence.
   # 3 looks like a reasonable number of clusters"""


# plt.clf()
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)  # Firstly call the algorithm
    kmeans.fit(versi_sc)  # fit is always used to train an algorithm
    wcss.append(kmeans.inertia_)  # inertia_ gives us the wcss value for each cluster.
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method',fontsize=20)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 2).fit(versi_sc)


####
# Test 1 - test to see if the anomalies are far from the cluster centroids
####

# Obtain predictions and calculate distance from cluster centroid
versi_sc_clusters = kmeans.predict(versi_sc)
versi_sc_clusters_centers = kmeans.cluster_centers_
dist = [np.linalg.norm(x-y) for x, y in zip(versi_sc, versi_sc_clusters_centers[versi_sc_clusters])]

print(versi_sc_clusters)
print(dist)

# Create fraud predictions based on outliers on clusters
km_y_pred = np.array(dist)
km_y_pred[dist >= np.percentile(dist, 95)] = 1
km_y_pred[dist < np.percentile(dist, 95)] = 0


# The anomalies flagged using distances from the centroid are not the mis-labelled cases. As you will see
# in test 2 this is because one of the three clusters contain only the mis-labelled cases.

####
# Test 2 - Testing to see if one of the clusters contain only the mis-labelled cases
####

# Versicolor dataframe with the clusters
versi_clus = pd.concat([versi_df,
                        pd.DataFrame(versi_sc_clusters,columns=['Clusters'])],axis=1)

# We can see that one of the clusters contain only the mis-labelled cases
# plt.clf()
plt.subplot(1,2,1)
sns.scatterplot(data=versi_clus, x='sepal_length', y='sepal_width', hue='Clusters', palette='deep')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend( loc='lower right')
plt.title('Sepal length v width')

plt.subplot(1,2,2)
sns.scatterplot(data=versi_clus, x='petal_length', y='petal_width', hue='Clusters', palette='deep')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.legend(loc='lower right')
plt.title('Petal length v width')
plt.show()


##############################################################################################################
                                        # DBSCAN
##############################################################################################################

# Density based clustering method (DBSCAN) to detect anomalies.
# The advantage of DBSCAN is that you do not need to define the number of
# clusters beforehand. Also, DBSCAN can handle weirdly shaped data (i.e. non-convex) much
# better than K-means can. Similar to above we take the smallest clusters in the data and label those as anomalies.

# Initialize and fit the DBSCAN model
db = DBSCAN(eps=0.8, min_samples=1, n_jobs=-1).fit(versi_sc)

# Obtain the predicted labels and calculate number of clusters
pred_labels = db.labels_
n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)

# Print performance metrics for DBSCAN
# There are only two clusters
print('Estimated number of clusters: %d' % n_clusters)


versi_db = pd.concat([versi_clus,
                        pd.DataFrame(pred_labels,columns=['db_cluster'])],axis=1)

# Under both DBSCAN and K means clustering the first two cases are being flagged as anomalies
versi_db.head()