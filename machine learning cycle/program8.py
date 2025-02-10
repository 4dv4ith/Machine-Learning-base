import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Data
X = np.array([[1.713, 1.586], [0.180, 1.786], [0.353, 1.240], [0.940, 1.566], 
              [1.486, 0.759], [1.266, 1.106], [1.540, 0.419], [0.459, 1.799], 
              [0.773, 0.186]])

y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1])  # True classes

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Function to implement K-means clustering
def kmeans(X, k, max_iters=100):
    # Step 1: Initialize centroids randomly from the dataset
    np.random.seed(42)  # For reproducibility
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for i in range(max_iters):
        # Step 2: Assign each data point to the nearest centroid
        labels = []
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            label = np.argmin(distances)  # Index of the closest centroid
            labels.append(label)

        labels = np.array(labels)

        # Step 3: Recalculate centroids by taking the mean of assigned points
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # If the centroids don't change, break the loop
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return centroids, labels

# Run the K-means algorithm with 3 clusters
k = 2
centroids, predicted_labels = kmeans(X, k)

# Predict the cluster for the new data point (VAR1=0.906, VAR2=0.606)
new_point = np.array([0.906, 0.606])
distances = [euclidean_distance(new_point, centroid) for centroid in centroids]
predicted_cluster = np.argmin(distances)

# Map the predicted cluster to the actual class labels (0 or 1)
# Since K-means does not predict the class directly, we can use majority voting within the clusters to assign class labels.
# Calculate the majority class in each cluster
cluster_to_class = {}
for i in range(k):
    cluster_data = y_true[predicted_labels == i]
    majority_class = np.bincount(cluster_data).argmax()  # Get the majority class in the cluster
    cluster_to_class[i] = majority_class

# Now assign the predicted class for the new point
predicted_class = cluster_to_class[predicted_cluster]

# Now, we will calculate the precision, recall, and F1 score.
# The predicted labels will be the class assigned to each point based on the clusters.

# Generate the predicted labels for all points in the dataset
final_predicted_labels = np.array([cluster_to_class[label] for label in predicted_labels])

# Calculate Precision, Recall, and F1 Score
precision = precision_score(y_true, final_predicted_labels)
recall = recall_score(y_true, final_predicted_labels)
f1 = f1_score(y_true, final_predicted_labels)

# Print results
print(f"Predicted class for (VAR1=0.906, VAR2=0.606): {predicted_class}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
