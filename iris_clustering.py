import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Iris Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ---------------- K-MEANS CLUSTERING ----------------

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

predicted_clusters = kmeans.fit_predict(df)

print("\nPredicted Clusters:")
print(predicted_clusters[:20])


# True labels
true_labels = iris.target

print("\nTrue Labels:")
print(true_labels[:20])


# Compare true labels and predicted clusters
comparison = pd.DataFrame({
    "True Label": true_labels,
    "Predicted Cluster": predicted_clusters
})

print("\nTrue Label vs Predicted Cluster:")
print(comparison.head(20))

print("\nComparison Table:")
print(pd.crosstab(
    comparison["True Label"],
    comparison["Predicted Cluster"]
))


# K-Means visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    df["sepal length (cm)"],
    df["sepal width (cm)"],
    c=predicted_clusters
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Clustering using K-Means")


# ---------------- PCA ----------------

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Reduce 4 dimensions to 2 dimensions
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

print("\nPCA Result:")
print(pca_data[:5])

print("\nPCA Shape:")
print(pca_data.shape)


# PCA visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=predicted_clusters
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Iris Dataset - PCA Visualization")


# Show both graphs
plt.show()
# ---------------- ELBOW METHOD ----------------

inertia = []

for k in range(1, 11):
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_model.fit(df)
    inertia.append(kmeans_model.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 6))

plt.plot(range(1, 11), inertia, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")

plt.show()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()