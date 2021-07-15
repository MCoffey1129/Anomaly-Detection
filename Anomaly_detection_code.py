
# Import packages
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


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

# Rename the first and fourth species as versicolor
iris.iloc[0,4] = 'versicolor'
iris.iloc[3,4] = 'versicolor'
iris.head() # the first two species have been updated


versi_df =  iris.loc[iris['species']=='versicolor']
versi_df = versi_df.reset_index(drop=True)
versi_df.head()


# Looking at the graphs it looks like they are anomalies/misclassifications
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


versi_red = versi_df.iloc[:,:4]

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)  # Firstly call the algorithm
    kmeans.fit(versi_red)  # fit is always used to train an algorithm
    wcss.append(kmeans.inertia_)  # inertia_ gives us the wcss value for each cluster.
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method',fontsize=20)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 2).fit(versi_red)

# Obtain predictions and calculate distance from cluster centroid
versi_red_clusters = kmeans.predict(versi_red)
versi_red_clusters_centers = kmeans.cluster_centers_
dist = [np.linalg.norm(x-y) for x, y in zip(versi_red, versi_red_clusters_centers[versi_red_clusters])]

print(versi_red_clusters_centers)

# Create fraud predictions based on outliers on clusters
km_y_pred = np.array(dist)
km_y_pred[dist >= np.percentile(dist, 95)] = 1
km_y_pred[dist < np.percentile(dist, 95)] = 0
