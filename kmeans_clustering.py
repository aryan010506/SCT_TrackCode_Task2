import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# STEP 1: Load the dataset
data = pd.read_csv("Mall_Customers.csv")
print("First 5 rows of the dataset:")
print(data.head())

# STEP 2: Select relevant features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# STEP 3: Use the Elbow Method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# STEP 4: Plot the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# STEP 5: Apply KMeans with the chosen number of clusters (e.g., 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add the cluster labels to the original data
data['Cluster'] = y_kmeans

# STEP 6: Visualize the clusters
plt.figure(figsize=(8, 5))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for i in range(5):
    plt.scatter(X[y_kmeans == i]['Annual Income (k$)'],
                X[y_kmeans == i]['Spending Score (1-100)'],
                s=100, c=colors[i], label=f'Cluster {i+1}')

# Plot cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', marker='*', label='Centroids')

plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# STEP 7: Save the final dataset with clusters (optional)
data.to_csv("Mall_Customers_Clustered.csv", index=False)
print("Clustering complete. File saved as Mall_Customers_Clustered.csv")
