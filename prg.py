
# ============================================================
# Mini Project 3: Iris Flower Clustering Project
# K-Means Clustering + PCA Visualization
# ============================================================

# ------------------------------------------------------------
# 1. Import required libraries
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


# ------------------------------------------------------------
# 2. Load the Iris Dataset
# ------------------------------------------------------------

# Dataset Name: Iris Flower Dataset
iris = load_iris()

# Input features
X = iris.data

# True labels
y = iris.target

# Flower names
target_names = iris.target_names

print("Dataset Name: Iris Flower Dataset")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Classes:", target_names)


# ------------------------------------------------------------
# 3. Display the dataset
# ------------------------------------------------------------

df = pd.DataFrame(X, columns=iris.feature_names)

print("\nFirst 5 rows of the dataset:")
print(df.head())


# ------------------------------------------------------------
# 4. Standardize the data
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nOriginal Data Shape:")
print(X.shape)

print("Standardized Data Shape:")
print(X_scaled.shape)


# ------------------------------------------------------------
# 5. Apply PCA
# ------------------------------------------------------------

# Reduce 4 dimensions to 2 dimensions
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nAfter PCA:")
print("Reduced Data Shape:", X_pca.shape)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "Total Variance Explained:",
    sum(pca.explained_variance_ratio_)
)


# ------------------------------------------------------------
# 6. Apply K-Means Clustering
# ------------------------------------------------------------

# Number of clusters = 3
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train K-Means using PCA data
predicted_clusters = kmeans.fit_predict(X_pca)


# ------------------------------------------------------------
# 7. Add results to DataFrame
# ------------------------------------------------------------

df["True_Label"] = y
df["Predicted_Cluster"] = predicted_clusters

print("\nTrue Labels vs Predicted Clusters:")
print(
    df[
        [
            "True_Label",
            "Predicted_Cluster"
        ]
    ].head(20)
)


# ------------------------------------------------------------
# 8. Display cluster centers
# ------------------------------------------------------------

print("\nK-Means Cluster Centers:")
print(kmeans.cluster_centers_)


# ------------------------------------------------------------
# 9. Visualize K-Means Clusters after PCA
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=predicted_clusters,
    cmap="viridis",
    s=60
)

# Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200,
    color="black",
    label="Cluster Centers"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "Iris Dataset - K-Means Clustering after PCA"
)

plt.legend()
plt.show()


# ------------------------------------------------------------
# 10. Visualize True Iris Labels
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    s=60
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "Iris Dataset - True Labels after PCA"
)

plt.colorbar(label="True Class")

plt.show()


# ------------------------------------------------------------
# 11. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y, predicted_clusters)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 12. Final Comparison
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "True Label": y,
    "Predicted Cluster": predicted_clusters
})

print("\nComplete Comparison:")
print(comparison)


# ------------------------------------------------------------
# 13. Project Summary
# ------------------------------------------------------------

print("\n======================================")
print("        PROJECT COMPLETED")
print("======================================")

print("Dataset       : Iris Flower Dataset")
print("Clustering    : K-Means")
print("Number of K   : 3")
print("Dimensionality: 4 → 2 using PCA")
print("Visualization : PCA + K-Means")
print("Comparison    : True Labels vs Clusters")

