def load_dataset(input_path = "dbs/cooked/") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset from the specified path.
    """
    test_data = np.load(input_path + "test_data.npz")
    X_te = test_data['X_test']
    y_te = test_data['y_test']
    
    train_data = np.load(input_path + "train_data.npz")
    X_tr = train_data['X_train']
    y_tr = train_data['y_train']
    return X_tr, X_te, y_tr, y_te