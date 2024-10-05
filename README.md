# DynoGraph

Source code for paper "DynoGraph: Dynamic Graph Construction for Nonlinear Dimensionality Reduction".

## Requirements

Before running the code, make sure you have Python 3.9 installed along with the necessary libraries. Install all required dependencies by running: 

```bash
pip install -r requirements.txt
```

## Datasets

The `data` folder includes both synthetic and real datasets used throughout the paper. 

- **Synthetic Dataset:**
  - s_curve_with_hole
- **Real Datasets:**
  - WarpPIE10P：Available from [Feature Selection Datasets](https://jundongl.github.io/scikit-feature/datasets.html)
  - COIL20：Available from [COIL20](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)
  - LandsatSatellite：Available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite)
  - HAR (Human Activity Recognition Using Smartphones): Available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
  - Fashion-MNIST：Available from [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
  - MNIST：Available from [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Data normalization:**
  - For image datasets, we implement uniform scaling by dividing all pixel values by the maximum value of 255. This approach effectively normalizes the entire dataset to a range of 0 to 1, ensuring that the relative intensity of each pixel remains constant throughout the image.
  - For synthetic s_curve_with_hole dataset , we use `sklearn.preprocessing.StandardScaler` for standardization normalization. 
  - HAR dataset is already appropriately scaled and requires no further normalization.

## Comparison Algorithms

To facilitate a thorough evaluation, DynoGraph compares its performance against several well-known dimensionality reduction algorithms, utilizing default parameters for each to ensure a fair and consistent comparison:

- **PCA**: Principal Component Analysis using `sklearn.decomposition.PCA`.
- **MDS**: Multidimensional Scaling using `sklearn.manifold.MDS`.
- **LLE**: Locally Linear Embedding using  `sklearn.manifold.locally_linear_embedding`.
- **Eigenmaps**: Laplacian Eigenmaps using `sklearn.manifold.SpectralEmbedding`.
- **t-SNE**: T-distributed Stochastic Neighbor Embedding using `sklearn.manifold.TSNE`.
- **UMAP**: Uniform Manifold Approximation and Projection, available at [UMAP GitHub Repository](https://github.com/lmcinnes/umap).
- **LargeVis**: Large-scale Visualization, available at [LargeVis GitHub Repository](https://github.com/lferry007/LargeVis).
- **TriMap**: TriMap, available at [TriMap GitHub Repository](https://github.com/eamid/trimap).
- **SpaceMAP**: SpaceMAP, available at [SpaceMAP GitHub Repository](https://github.com/zuxinrui/SpaceMAP).

## Quick Start

#### 1. `DynoGraph.py`

- **Description**: This file contains the source code for `DynoGraph`, the main algorithm introduced in our paper. 

- **Usage**:

  ```
  from DynoGraph import DynoGraph
  
  embedding = DynoGraph(n_components=2).fit_transform(data)
  ```

#### 2. `test_s_curve_with_hole.py`

- **Description**: This script is designed to test and visualize the performance of DynoGraph and all baselines on the synthetic dataset 's_curve_with_hole'. It includes a function, `make_s_curve_with_hole_uniform`, which generates a uniformly distributed rectangular and removes a hole in the center before folding it into a three-dimensional S-curve manifold.

- **Outputs**:

  - Visualizations of the original 3D data, 2D ground truth, and embeddings from each algorithm.
  - Calculated Procrustes disparity between the 2D ground truth and each embedding to quantify alignment accuracy.

- **How to Run**:

  ```
  python test_s_curve_with_hole.py
  ```

#### 3. `test_real_data.py`

- **Description**: This script evaluates the performance of DynoGraph and all baselines on all real datasets. 

- **Outputs**:

  - Visualizations of embedding results on real datasets.
  - Computed Adjusted Mutual Information (AMI) scores for k-means clustering.
  - Accuracy scores for k-NN classifier, with k values ranging from 1% to 20% of the dataset size, to ensure robustness across different scales of data.
  - Average performance metrics, standard deviations, and computational runtimes for each algorithm across 10 runs.

- **How to Run**:

  ```
  python test_real_data.py
  ```
