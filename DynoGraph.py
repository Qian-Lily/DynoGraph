from sklearn.base import BaseEstimator
import numpy as np
from numba import njit, prange
from math import erf
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from sklearn.utils import check_random_state
from scipy.sparse import coo_matrix
import faiss


IS2 = np.float32(0.7071067811865475)  # 1 / np.sqrt(2)
IS2PL2 = np.float32(0.5755520495360813)  # 1 / (np.sqrt(2 * np.pi) * np.log(2))
SP = np.float32(1.7724538509055159)  # np.sqrt(np.pi)
LN2 = np.float32(0.6931471805599453)  # np.log(2)
ISPL2 = np.float32(0.8139535143055576)  # 1 / (np.sqrt(np.pi) * np.log(2))
I2PL2 = np.float32(0.9184481885265703)  # 2 / (np.pi * np.log(2))


@njit("f8(f4[:],f4[:])")
def euclid_dist(x1, x2):
    """
    Euclidean distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i] - x2[i]) ** 2
    return np.sqrt(result)


@njit('f4[:](f4[:,:])', parallel=True)
def calculate_distance_array(X):
    n = X.shape[0]
    dist_array = np.empty(n * (n - 1) // 2, dtype=np.float32)  # Array to hold upper triangular distances
    for i in prange(n - 1):
        for j in range(i + 1, n):
            idx = i * n - (i * (i + 1) // 2) + (j - i - 1)  # Calculate the index in the flattened array
            dist_array[idx] = euclid_dist(X[i], X[j])
    return dist_array


@njit('Tuple((i1[:,:], f4, f4))(f4[:,:], i8[:,:], f4[:], i4)', parallel=True)
def calculate_adjacency_matrix_knn(knn_dists, knn_indices, epsilon_dist, k_neighbors):
    n = knn_dists.shape[0]
    adj_matrix = np.zeros((n, n), dtype=np.int8)

    # Arrays to store intermediate max/min values
    max_dist = np.zeros(n, dtype=np.float32)
    min_dist = np.full(n, np.inf, dtype=np.float32)

    for i in prange(n):
        # Connect at least min_neighbors
        for j_index in range(k_neighbors):
            j = knn_indices[i, j_index]
            if i == j:
                continue
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

        max_dist[i] = knn_dists[i, k_neighbors - 1]

        # Further refine connections based on epsilon distance
        for j_index in range(k_neighbors, knn_indices.shape[1]):
            j = knn_indices[i, j_index]
            dist_ij = knn_dists[i, j_index]
            if dist_ij <= min(epsilon_dist[i], epsilon_dist[j]):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
                max_dist[i] = dist_ij
            elif dist_ij > epsilon_dist[i]:
                if min_dist[i] == np.inf:
                    min_dist[i] = dist_ij
                break
            else:
                if min_dist[i] == np.inf:
                    min_dist[i] = dist_ij

    max_dist = np.max(max_dist)
    min_dist = np.min(min_dist)

    return adj_matrix, max_dist, min_dist


@njit('Tuple((i4, i4))(f4[:, :], i8[:, :], f4, f4)', parallel=True)
def find_positions_knn(knn_dists, knn_indices, max_distance, min_distance):
    n = knn_dists.shape[0]

    # Arrays to store counts for each thread
    max_counts = np.zeros(n, dtype=np.int32)
    min_counts = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        check_min_distance = True
        for j_index in range(knn_dists.shape[1]):
            j = knn_indices[i, j_index]
            if j > i:
                dist_ij = knn_dists[i, j_index]
                if dist_ij > max_distance:
                    break
                if dist_ij <= max_distance:
                    max_counts[i] += 1
                    if check_min_distance and dist_ij <= min_distance:
                        min_counts[i] += 1
                    elif dist_ij > min_distance:
                        check_min_distance = False

    # Summing the arrays to get the final counts
    max_index = np.sum(max_counts)
    min_index = np.sum(min_counts)

    return max_index, min_index


@njit("Tuple((f4[:,:], i4[:, :, :]))(f4[:,:], i8[:,:], i4, i4[:])", parallel=True)
def find_min_edges_knn(knn_dists, knn_indices, components, labels):
    min_edge_indices = np.full((components, components, 2), -1, dtype=np.int32)
    new_graph = np.full((components, components), np.inf, dtype=np.float32)

    for i in range(components):
        indices_i = np.flatnonzero(labels == i)
        for j in range(i + 1, components):
            for index_i in indices_i:
                for k, index_j in enumerate(knn_indices[index_i]):
                    if labels[index_j] == j:
                        if knn_dists[index_i, k] < new_graph[i, j]:
                            new_graph[i, j] = knn_dists[index_i, k]
                            min_edge_indices[i, j] = [index_i, index_j]
                        break

    return new_graph, min_edge_indices


def connect_components_using_mst_knn(knn_dists, knn_indices, adj_matrix, components, labels):
    """
    Connect disconnected components using Minimum Spanning Tree (MST).
    """
    new_graph, min_edge_indices = find_min_edges_knn(knn_dists, knn_indices, components, labels)

    mst = minimum_spanning_tree(new_graph).tocoo()

    mst_edges = min_edge_indices[mst.row, mst.col]
    adj_matrix[mst_edges[:, 0], mst_edges[:, 1]] = 2
    adj_matrix[mst_edges[:, 1], mst_edges[:, 0]] = 2

    components_sub, labels_sub = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    if components_sub > 1:
        np.random.seed(42)
        # Get unique labels for each component
        for component_label in range(components_sub - 1):
            component_a = np.flatnonzero(labels_sub == component_label)
            component_b = np.flatnonzero(labels_sub == component_label + 1)
            point_a = np.random.choice(component_a)
            point_b = np.random.choice(component_b)

            adj_matrix[point_a, point_b] = adj_matrix[point_b, point_a] = 2

    return adj_matrix


def dynamic_knn_mst(X):
    n = X.shape[0]
    n_neighbors = int(np.sqrt(n))
    n_neighbors_large = min(n, 3 * n_neighbors, 2048)
    min_neighbors = 5

    gpures = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    index = faiss.GpuIndexFlatL2(gpures, X.shape[1], flat_config)
    index.add(X.astype('float32'))
    knn_dists, knn_indices = index.search(X, n_neighbors_large)
    knn_dists = np.sqrt(knn_dists)

    epsilon_dist = np.mean(knn_dists[:, 1:n_neighbors + 1], axis=1)
    adj_matrix, max_dist, min_dist = calculate_adjacency_matrix_knn(knn_dists, knn_indices, epsilon_dist, min_neighbors)
    max_index, min_index = find_positions_knn(knn_dists, knn_indices, max_dist, min_dist)
    components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

    if components > 1:
        adj_matrix = connect_components_using_mst_knn(knn_dists, knn_indices, adj_matrix, components, labels)

    return adj_matrix, min_index, max_index


@njit('i1[:,:](i1[:,:],i4[:,:],i4[:,:],i4)', parallel=True)
def create_sampling_matrix(adj_matrix, random_sampling, edges, n_edges):
    sampling_matrix = adj_matrix.copy()

    for n_edge in prange(n_edges):
        i, j = edges[n_edge]

        # Use pre-generated random numbers
        not_edge1, not_edge2 = random_sampling[n_edge]

        # Check and update for the first negative sample
        if adj_matrix[i, not_edge1] == 0 and i != not_edge1:
            sampling_matrix[i, not_edge1] = -1
            sampling_matrix[not_edge1, i] = -1

        # Check and update for the second negative sample
        if adj_matrix[j, not_edge2] == 0 and j != not_edge2:
            sampling_matrix[j, not_edge2] = -1
            sampling_matrix[not_edge2, j] = -1

    return sampling_matrix


@njit('Tuple((f4,f4))(f4)')
def update_edge(dist):
    h = dist * IS2
    if h < 2.44459:
        EXP = np.exp(-h * h)
        ERF = erf(h)
        ERF -= 1.0
        ERF = min(ERF, -1e-20)
        quot = EXP / ERF
    else:
        quot = - SP * h

    w = (ISPL2 * h + I2PL2 * quot) * quot
    w = max(w, 0)

    d = dist
    if w != 0:
        d += IS2PL2 * quot / w
    else:
        d = 0
    return d, w


@njit('Tuple((f4,f4))(f4)')
def update_no_edge(dist):
    h = dist * IS2
    if h > -2.44459:
        EXP = np.exp(-h * h)
        ERF = erf(h)
        ERF += 1.0
        ERF = max(ERF, 1e-20)
        quot = EXP / ERF
    else:
        quot = - SP * h

    w = (ISPL2 * h + I2PL2 * quot) * quot
    w = max(w, 0)

    d = dist
    if w != 0:
        d += IS2PL2 * quot / w
    else:
        d = 0
    return d, w


@njit('f4[:,:](i1[:,:],f4[:,:])', parallel=True)
def update_coord(adj_matrix, coord):
    n = coord.shape[0]
    positions = np.zeros_like(coord, dtype=np.float32)

    for i in prange(n):
        sum_weights = 0
        for j in range(n):
            if adj_matrix[i, j] == 0:
                continue

            sij = euclid_dist(coord[i], coord[j])

            if adj_matrix[i, j] == -1:
                d, w = update_no_edge(sij)
            else:
                d, w = update_edge(sij)

            sum_weights += w

            if sij != 0:
                sij = d / sij

            positions[i] += w * (coord[j] + sij * (coord[i] - coord[j]))

        positions[i] /= sum_weights

    return positions


@njit("i1[:,:](i1[:,:],f4[:,:],i4,i4,i4)")
def modify_edges(adj_matrix, coord, min_index, max_index, iter_graph):
    n = adj_matrix.shape[0]
    adj_matrix_modi = adj_matrix.copy()
    # Calculate the Euclidean distances between all pairs of nodes
    dist_array = calculate_distance_array(coord)

    # Find the threshold values for edge addition and deletion
    T_far = np.partition(dist_array, max_index)[max_index]
    T_near = np.partition(dist_array, min_index)[min_index]

    np.random.seed(iter_graph)

    neighbors = [np.nonzero(adj_matrix_modi[i])[0] for i in range(n)]

    # count_addition = 0
    # count_deletion = 0

    idx = 0
    # Iterate over all pairs of nodes, considering only the upper triangle to avoid repetition
    for i in range(n):
        for j in range(i + 1, n):
            # Attempt to add an edge if it exists and the distance exceeds T_far
            if adj_matrix[i, j] == 1 and dist_array[idx] > T_far:
                N_i = neighbors[i]
                N_j = neighbors[j]
                N_common = np.intersect1d(N_i, N_j)

                N_nc_i = np.array([x for x in N_i if x not in N_common and x != j])
                N_nc_j = np.array([x for x in N_j if x not in N_common and x != i])

                # Try to add new edges if there are non-common neighbors
                if N_nc_i.size > 0 and N_nc_j.size > 0:
                    candidate_edges = np.array([(i, nbr) for nbr in N_nc_j] + [(j, nbr) for nbr in N_nc_i])
                    edge_to_add = candidate_edges[np.random.randint(candidate_edges.shape[0])]
                    adj_matrix_modi[edge_to_add[0], edge_to_add[1]] = 1
                    adj_matrix_modi[edge_to_add[1], edge_to_add[0]] = 1
                    # count_addition += 1

                elif N_nc_i.size == 0 and N_nc_j.size > 0:
                    candidate_edges = np.array([(i, nbr) for nbr in N_nc_j])
                    edge_to_add = candidate_edges[np.random.randint(candidate_edges.shape[0])]
                    adj_matrix_modi[edge_to_add[0], edge_to_add[1]] = 1
                    adj_matrix_modi[edge_to_add[1], edge_to_add[0]] = 1
                    # count_addition += 1

                elif N_nc_j.size == 0 and N_nc_i.size > 0:
                    candidate_edges = np.array([(j, nbr) for nbr in N_nc_i])
                    edge_to_add = candidate_edges[np.random.randint(candidate_edges.shape[0])]
                    adj_matrix_modi[edge_to_add[0], edge_to_add[1]] = 1
                    adj_matrix_modi[edge_to_add[1], edge_to_add[0]] = 1
                    # count_addition += 1

            elif dist_array[idx] < T_near and adj_matrix[i, j] == 0:
                N_i = np.nonzero(adj_matrix_modi[i])[0]
                N_j = np.nonzero(adj_matrix_modi[j])[0]
                N_common = np.intersect1d(N_i, N_j)
                if len(N_common) > 1:
                    candidate_edges = np.array([(i, nbr) for nbr in N_common] + [(j, nbr) for nbr in N_common])
                    edge_to_delete = candidate_edges[np.random.randint(candidate_edges.shape[0])]
                    adj_matrix_modi[edge_to_delete[0], edge_to_delete[1]] = 0
                    adj_matrix_modi[edge_to_delete[1], edge_to_delete[0]] = 0
                    # count_deletion += 1

            idx += 1

    # print(f"Additions: {count_addition}, Deletions: {count_deletion}")
    return adj_matrix_modi


class DynoGraph(BaseEstimator):
    def __init__(self,
                 n_components=2,
                 random_state=None,
                 iter_embedding=200,
                 iter_graph=2
                 ):
        self.n_components = n_components
        self.random_state = random_state
        self.iter_embedding = iter_embedding
        self.iter_graph = iter_graph

    def fit(self, X):
        X = X.astype(np.float32)

        # start = time.time()
        self._adj_matrix, self._min_index, self._max_index = dynamic_knn_mst(X)
        # print("Graph Time:", round(time.time() - start, 2))

        # start = time.time()
        n_samples = X.shape[0]
        random_state = check_random_state(self.random_state)
        self.embedding_ = random_state.uniform(low=0.0, high=1.0, size=(n_samples, self.n_components)).astype(
            np.float32)

        upper_triangle_matrix = np.triu(self._adj_matrix)
        sparse_matrix = coo_matrix(upper_triangle_matrix)
        edges = np.array([sparse_matrix.row, sparse_matrix.col]).T

        n_edges = edges.shape[0]

        for i_init in range(self.iter_embedding):
            np.random.seed(i_init)
            random_sampling = np.random.randint(0, n_samples - 1, size=(n_edges, 2))
            sampling_matrix = create_sampling_matrix(self._adj_matrix, random_sampling, edges, n_edges)
            self.embedding_ = update_coord(sampling_matrix, self.embedding_)

        for i_graph in range(self.iter_graph):

            adj_matrix_modi = modify_edges(self._adj_matrix, self.embedding_, self._min_index, self._max_index, i_graph)

            upper_triangle_matrix_modi = np.triu(adj_matrix_modi)
            sparse_matrix_modi = coo_matrix(upper_triangle_matrix_modi)
            edges_modi = np.array([sparse_matrix_modi.row, sparse_matrix_modi.col]).T

            n_edges_modi = edges_modi.shape[0]

            for i_modi in range(self.iter_embedding):
                np.random.seed(i_modi + 42)
                random_sampling = np.random.randint(0, n_samples - 1, size=(n_edges_modi, 2))
                sampling_matrix = create_sampling_matrix(adj_matrix_modi, random_sampling, edges_modi, n_edges_modi)
                self.embedding_ = update_coord(sampling_matrix, self.embedding_)

        # print("Embedding Time:", round(time.time() - start, 2))
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_
