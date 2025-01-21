from sklearn.decomposition import PCA

def apply_pca(features, n_components=10):
    # Initialize PCA
    pca = PCA(n_components=n_components)
    """
    Fit: calculates the principal components based on the input data
    Transform: projects the original features onto the new space defined by the principal components
    """
    transformed_features = pca.fit_transform(features)
    return transformed_features