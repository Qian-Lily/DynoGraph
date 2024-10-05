import numpy as np
import matplotlib.pyplot as plt
from DynoGraph import DynoGraph
from sklearn import manifold
from sklearn.decomposition import PCA
import umap
import trimap
from SpaceMAP._spacemap import SpaceMAP
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time
import pandas as pd


# dataname = 'WarpPIE10P'
# dataset = np.loadtxt(f'./data/{dataname}/WarpPIE10P.txt')
# data = dataset[:, 1:]  # All rows, all but first column for data
# labels = dataset[:, 0]  # All rows, first column for labels
#
# # Normalize the data to the range [0, 1]
# data_normalized = data / 255


dataname = 'COIL20'
# Load data and labels from the COIL20 dataset
dataset = np.loadtxt(f'./data/{dataname}/COIL20.txt')
data = dataset[:, :-1]  # All rows, all but last column for data
labels = dataset[:, -1]  # All rows, last column for labels

# Normalize the data to the range [0, 1]
data_normalized = data / 255


# dataname = 'Landsat'
# dataset = np.loadtxt(f'./data/{dataname}/LandsatSatellite_Statlog.txt')
# data = dataset[:, :-1]  # All rows, all but last column for data
# labels = dataset[:, -1]  # All rows, last column for labels
#
# # Normalize the data to the range [0, 1]
# data_normalized = data / 255


# dataname = 'HAR'
# dataset = np.loadtxt(f'./data/{dataname}/Human_Activity_Recognition_Using_Smartphones.txt')
# data = dataset[:, :-1]  # All rows, all but last column for data
# labels = dataset[:, -1]  # All rows, last column for labels
#
# # # Note: The features with values ranging between -1 and 1, thus no normalization is needed.
# data_normalized = data


# dataname = 'Fashion_MNIST'
# dataset = np.loadtxt(f'./data/{dataname}/Fashion_MNIST.txt')
# data = dataset[:, :-1]  # All rows, all but last column for data
# labels = dataset[:, -1]  # All rows, last column for labels
#
# # Normalize the data to the range [0, 1]
# data_normalized = data / 255


# dataname = 'MNIST'
# dataset = np.loadtxt(f'./data/{dataname}/MNIST.txt')
# data = dataset[:, :-1]  # All rows, all but last column for data
# labels = dataset[:, -1]  # All rows, last column for labels
#
# # Normalize the data to the range [0, 1]
# data_normalized = data / 255

data = data_normalized
data_size = len(data)
k_values = [int(0.01 * i * data_size) for i in range(1, 21)]
results = pd.DataFrame()

# methods = ["DynoGraph", "TSNE", "UMAP", "SpaceMAP", "TriMap", "LargeVis", "Eigenmaps", "PCA", "LLE", "MDS"]
methods = ["DynoGraph"]


skip_conditions = {
    "Fashion_MNIST": ["MDS", "Eigenmaps"],  # Out of memory (>64GB)
    "MNIST": ["MDS", "Eigenmaps"]  # Out of memory (>64GB)
}

for method in methods:
    if method in skip_conditions.get(dataname, []):
        continue
    print(f"Processing method: {method}")

    for i in range(10):
        print(f"Processing iteration: {i}")
        random_seed = i

        start_time = time.time()

        if method == "LLE":
            embedding = manifold.LocallyLinearEmbedding(random_state=random_seed).fit_transform(data)
        elif method == "MDS":
            embedding = manifold.MDS(random_state=random_seed).fit_transform(data)
        elif method == "Eigenmaps":
            embedding = manifold.SpectralEmbedding(random_state=random_seed).fit_transform(data)
        elif method == "TSNE":
            embedding = manifold.TSNE(random_state=random_seed).fit_transform(data)
        elif method == 'PCA':
            embedding = PCA(n_components=2, random_state=random_seed).fit_transform(data)
        elif method == 'UMAP':
            embedding = umap.UMAP(random_state=random_seed).fit_transform(data)
        elif method == 'TriMap':
            embedding = trimap.TRIMAP().fit_transform(data)
            # np.save(f'./data/{dataname}/{dataname}_3d_{method}_rs{i}.npy', embedding)
            # embedding = np.load(f'./data/{dataname}/{dataname}_3d_{method}_rs{i}.npy')
        elif method == 'SpaceMAP':
            data = data.astype('float32')
            embedding = SpaceMAP().fit_transform(data)
        elif method == 'LargeVis':
            embedding = np.loadtxt(f'./data/{dataname}/{dataname}_3d_{method}_rs{i}.txt')
        elif method == 'DynoGraph':
            embedding = DynoGraph(random_state=random_seed).fit_transform(data)

        end_time = time.time()
        duration = end_time - start_time

        # Visualize the embedding when random_seed is 0
        if random_seed == 0:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=1)
            plt.xticks([])
            plt.yticks([])
            plt.title(f'{dataname} - {method}', fontsize=20)
            plt.show()

        # Perform KMeans clustering on the reduced data to group similar instances
        n_clusters = len(np.unique(labels))  # Determine the number of unique labels as the number of clusters
        predicted_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(embedding)
        # Calculate AMI (Adjusted Mutual Information) to evaluate the clustering performance
        ami_score = adjusted_mutual_info_score(labels, predicted_labels)

        knn_results = {}
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            # Apply Stratified k-Fold cross-validation to evaluate the k-NN classifier on the reduced data
            cv = StratifiedKFold(n_splits=10)  # Using 10 folds
            # Evaluate k-NN classifier accuracy using cross-validation
            knn_scores = cross_val_score(knn, embedding, labels, cv=cv)
            knn_results[f'k_{k}'] = np.mean(knn_scores)

        result_entry = {
            "Dataset": dataname,
            "Method": method,
            "Iteration": i,
            "Duration": duration,
            "AMI": ami_score,
            **knn_results
        }
        results = results.append(result_entry, ignore_index=True)

    results.drop(columns=['Iteration'], inplace=True)
    results_statistic = results.groupby('Method').agg(['mean', 'std'])
    print(results_statistic)
    results_statistic.to_csv(f'{dataname}_results.csv')
