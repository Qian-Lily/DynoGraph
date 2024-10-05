import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
import umap
from DynoGraph import DynoGraph
import trimap
from SpaceMAP._spacemap import SpaceMAP
from scipy.spatial import procrustes
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


# # Generate synthetic data
# def make_s_curve_with_hole_uniform(n_samples=100, noise=0.0, random_state=None):
#     """Generate a 3D S-curve dataset with a hole in the center."""
#
#     generator = check_random_state(random_state)
#
#     t = 3 * np.pi * (generator.uniform(size=(1, n_samples)) - 0.5)
#     X = np.empty(shape=(n_samples, 3), dtype=np.float64)
#     X[:, 0] = np.sin(t)
#     X[:, 1] = 2.0 * generator.uniform(size=n_samples)
#     X[:, 2] = np.sign(t) * (np.cos(t) - 1)
#     X += noise * generator.standard_normal(size=(3, n_samples)).T
#     t = np.squeeze(t)
#
#     X_orig = np.empty(shape=(n_samples, 2), dtype=np.float64)
#     X_orig[:, 0] = t
#     X_orig[:, 1] = X[:, 1]
#
#     # Define the anchor point
#     anchor = np.array([0, 1, 0])
#
#     # Calculate the squared Euclidean distance from the anchor point
#     indices = np.sum((X - anchor) ** 2, axis=1) > 0.3
#     labels = t[indices]
#
#     # Filter the points to create a hole in the S-curve
#     data = X[indices]
#
#     data_2d = X_orig[indices]
#
#     if noise > 0:
#         data += noise * random_state.normal(size=data.shape)
#
#     return data_2d, data, labels
#
#
# data_2d, data_3d, labels_3d = make_s_curve_with_hole_uniform(n_samples=9000, random_state=0)


# Load synthetic data from file
dataname = 's_curve_with_hole'

# Load 3D data and labels
data_3d = np.loadtxt(f'./data/{dataname}/{dataname}_3d_data.txt')
labels_3d = np.loadtxt(f'./data/{dataname}/{dataname}_3d_label.txt')

# Load 2D ground truth
data_2d = np.loadtxt(f'./data/{dataname}/{dataname}_2d_data.txt')
data = StandardScaler().fit_transform(data_3d)

# Plot 3D data
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')
scatter_3d = ax_3d.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels_3d, cmap='Spectral', s=10)
ax_3d.set_axis_off()
plt.title('3D Scurve_hole', fontsize=20)
plt.show()

# Plot 2D ground truth data
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_3d, cmap='Spectral', s=10)
plt.xticks([])
plt.yticks([])
# Rotate the plot 180 degrees horizontally
plt.xlim(plt.xlim()[::-1])
plt.ylim(plt.ylim()[::-1])
# Setting the aspect ratio of the plot to 'equal' to maintain the data's proportionality
plt.gca().set_aspect('equal', adjustable='datalim')
plt.title('Ground Truth', fontsize=20)
plt.show()

methods = ["DynoGraph", "TSNE", "UMAP", "SpaceMAP", "TriMap", "LargeVis", "Eigenmaps", "PCA", "LLE", "MDS"]

random_seed = 0

for method in methods:
    print(f"Processing method: {method}")

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
        np.save(f'./data/{dataname}/{dataname}_3d_{method}.npy', embedding)
        # embedding = np.load(f'./data/{dataname}/{dataname}_3d_{method}.npy')
    elif method == 'SpaceMAP':
        data = data.astype('float32')
        embedding = SpaceMAP().fit_transform(data)
    elif method == 'LargeVis':
        embedding = np.loadtxt(f'./data/{dataname}/{dataname}_3d_{method}.txt')
    elif method == 'DynoGraph':
        embedding = DynoGraph(random_state=random_seed).fit_transform(data)

    # Perform Procrustes analysis to compare the shape similarity between the original 2D ground truth and the embedding
    _, _, disparity = procrustes(data_2d, embedding)

    # Output the Procrustes disparity result
    print(f'Procrustes disparity between 2D ground truth and embedding: {disparity:.3f}')

    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_3d, cmap='Spectral', s=10)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{method} - disparity: {disparity:.3f}', fontsize=20)
    plt.show()
