import numpy as np
import pandas as pd
import torch
from enum import Enum
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class reduction_techniques(Enum):
    UMAP = "umap"
    PCA = "pca"
    TSNE = "tsne"

def load_df(
    input_path: str = "./dbs/intermittent/",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the dataframes from the specified directory. The directory should contain two csv files:
    `tr_df.csv` and `te_df.csv`. The function will load the dataframes and return them as a tuple.
    Args:
        input_path (str): The directory to load the dataframes from.
    returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing dataframes.
    """
    tr_df = pd.read_csv(input_path + "tr_df.csv")
    te_df = pd.read_csv(input_path + "te_df.csv")
    return tr_df, te_df

def get_tensor(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a DataFrame to two tensors: one for the features and one for the labels. This
    function will not handle the dimensionality of the tensors. The formatting of the dimensions
    to be compatible with LSTM models will be handled in `adjust_dimensions`.
    Args:
        df (pd.DataFrame): The DataFrame to be converted to a tensor.
                This can be either the training or testing DataFrame.
    Returns:
        X (np.ndarray): The tensor of features. with shape (n_samples, n_features).
        y (np.ndarray): The tensor of labels. with shape (n_samples, n_labels).
    """
    #extract label
    y = torch.tensor(df['spain'].values.reshape(-1, 1), dtype=torch.float32)
    #extract features
    X = torch.tensor(df.drop(columns=['spain']).values, dtype=torch.float32)

def reduce_feature_dimensions(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: reduction_techniques = reduction_techniques.UMAP,
    n_components: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduces the dimensionality of the features tensor. This is done by
    using a None linear dimensionality reduction technique. The technical
    suggestion is to use a UMAP model. Though the developer is incetivized
    to experiment with other models.
    Args:
        X_train (np.ndarray): The tensor of features for the training data.
        X_test (np.ndarray): The tensor of features for the testing data.
        method (reduction_techniques): The reduction technique to be used.
    Returns:
        X_train (np.ndarray): The tensor of features for the training data.
        X_test (np.ndarray): The tensor of features for the testing data.

    """
    if method == reduction_techniques.UMAP:
        
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        X_train = reducer.fit_transform(X_train)
        X_test = reducer.transform(X_test)
    elif method == reduction_techniques.PCA:
        
        pca = PCA(n_components=n_components, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    elif method == reduction_techniques.TSNE:
        
        tsne = TSNE(n_components=n_components, random_state=42)
        X_train = tsne.fit_transform(X_train)
        X_test = tsne.transform(X_test)
    return X_train, X_test

def save_tensors(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_path: str = "./dbs/cooked/",
) -> int:
    """
    Saves the tensors to a directory as numpy files `npz` (NOT `npy`).
    args:
        X_train (np.ndarray): The tensor of features for the training data.
        X_test (np.ndarray): The tensor of features for the testing data.
        y_train (np.ndarray): The tensor of labels for the training data.
        y_test (np.ndarray): The tensor of labels for the testing data.
        output_path (str): The directory to save the tensors.
    returns:
        int: 0 if successful and 1 if not.
    """
    try:
        np.savez_compressed(
            output_path + "train_data.npz",
            X_train=X_train,
            y_train=y_train,
        )
        np.savez_compressed(
            output_path + "test_data.npz",
            X_test=X_test,
            y_test=y_test,
        )
        return 0
    except Exception as e:
        print(f"Error saving tensors: {e}")
        return 1
    ...
