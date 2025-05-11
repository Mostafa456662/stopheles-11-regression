### ~~~ GLOBAL IMPORTS ~~~ ###
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from enum import Enum
import pandas as pd
import numpy as np
import torch
import umap


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
    # extract label
    y = torch.tensor(df["spain"].values.reshape(-1, 1), dtype=torch.float32)
    # extract features
    X = torch.tensor(df.drop(columns=["spain"]).values, dtype=torch.float32)

    # convert to numpy arrays
    X = X.numpy()
    y = y.numpy()

    return X, y


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


def create_timeseries_dataset(dataset, n_past, n_future):
    """
    Creates a timeseries dataset for training or testing a model.

    Args:
      dataset: The original dataset.
      n_past: The number of past time steps to use as input.
      n_future: The number of future time steps to predict.

    Returns:
      A tuple containing the input features (trainX) and the target values (trainY).
    """
    raise NotImplementedError(
        "The function `create_timeseries_dataset` is not implemented yet. "
        "Please implement the function to create a timeseries dataset."
    )


def adjust_dimensions(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Adjusts the dimensions of the tensors to be compatible with LSTM models.
    The developer needs to do a LOT of research on how to properly format
    the tensors for LSTM models. The developer is encouraged to read the
    documentation on the `tf.keras.layers.LSTM` layer. Furthermore the following
    video is a great resource: `https://www.youtube.com/watch?v=tepxdcepTbY`.
    Args:
        X_train (np.ndarray): The tensor of features for the training data.
        X_test (np.ndarray): The tensor of features for the testing data.
        y_train (np.ndarray): The tensor of labels for the training data.
        y_test (np.ndarray): The tensor of labels for the testing data.
        n_steps (int): The number of time steps to be used in the LSTM model.
    Returns:
        X_train (np.ndarray): The tensor of features for the training data.
            - shape of (batch, timesteps, feature)
        X_test (np.ndarray): The tensor of features for the testing data.
            - shape of (batch, timesteps, feature)
        y_train (np.ndarray): The tensor of labels for the training data.
        y_test (np.ndarray): The tensor of labels for the testing data.
    """
    raise NotImplementedError(
        "The function `adjust_dimensions` is not implemented yet. "
        "Please implement the function to adjust the dimensions of the tensors."
    )


def main() -> int:
    """ """
    ### init variables ###
    input_path = "./dbs/preprocessing/"
    reduction_method = reduction_techniques.PCA
    n_steps: int = 14

    ### load dataframes ###
    tr_df, te_df = load_df(input_path=input_path)

    ### convert to tensors ###
    X_tr, y_tr = get_tensor(tr_df)
    X_te, y_te = get_tensor(te_df)

    ### ~~~ EXPLORE ~~~ ###
    before_reduction_str: str = "before reduction"
    print(f"{before_reduction_str:=^40}")
    print(f"X_tr.dim = {X_tr.shape}")
    print(f"X_te.dim = {X_te.shape}")
    print(f"y_tr.dim = {y_tr.shape}")
    print(f"y_te.dim = {y_te.shape}")
    ### ~~~ EXPLORE ~~~ ###

    ### reduce dimensions ###
    target_dim: int = X_tr.shape[1] // 2
    X_tr, X_te = reduce_feature_dimensions(
        X_train=X_tr,
        X_test=X_te,
        method=reduction_method,
        n_components=target_dim,
    )

    ### ~~~ EXPLORE ~~~ ###
    after_reduction_str: str = "after reduction"
    print(f"{after_reduction_str:=^40}")
    print(f"X_tr.dim = {X_tr.shape}")
    print(f"X_te.dim = {X_te.shape}")
    print(f"y_tr.dim = {y_tr.shape}")
    print(f"y_te.dim = {y_te.shape}")
    ### ~~~ EXPLORE ~~~ ###

    ### adjust dimensions ###
    X_tr, X_te, y_tr, y_te = adjust_dimensions(
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        n_steps=n_steps,
    )

    return 0


if __name__ == "__main__":
    main()
