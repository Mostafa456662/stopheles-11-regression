### ~~~ GLOBAL IMPORTS ~~~ ###
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import joblib
import tqdm





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



def create_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    """
    Create the LSTM model with the specified input shape.
    returns: "tf.keras.Model" the model
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray) -> tf.keras.callbacks.History:
    """
    Train the model with the given training data.
    returns: "tf.keras.callbacks.History" the history of the training
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
   
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    return history
    

    
def test_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Test the model with the given test data.
    """
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test MAE: {mae}")


def plot_history(history: tf.keras.callbacks.History) -> None:
    """
    Plot the training history.
    """
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plot_predictions(model, X_test, y_test, y_scaler) -> None:
    """
    Plot the last 50 predictions of the model.
    """
    y_pred = model.predict(X_test)

    y_pred_orig = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Get the last 50 predictions and actual values
    y_pred_last = y_pred_orig[:50]
    y_test_last = y_test_orig[:50]

    plt.plot(y_test_last, label='Actual')
    plt.plot(y_pred_last, label='Predicted')
    plt.legend()
    plt.title("Last 50 Predictions vs Actual (Original Scale)")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_model(model: tf.keras.Model, input_path = "src/models/my_model.keras") -> None:
    """
    Save the model to the specified path.
    """
    model.save(input_path)

def main() -> int:
    """
    do the following:
    1. load the dataset
    2. construct the model
        - n_lstm layers with the correct configs
        - n_dense layers
        - (i need dropout layer between every lstm)
        - (i need last lstm to return sequences)
    3. compile the model
        - loss function
        - optimizer
        - metrics
    4. train the model
    5. test the model
    6. show graph of history
    7. save the model
    """
    # Load the scaler
    y_scaler = joblib.load("src/data/scalers/scalers.pkl")

    pbar = tqdm.tqdm(total = 7)
    # Load the dataset
    pbar.set_description("Loading dataset")
    X_tr, X_te, y_tr, y_te = load_dataset()
    pbar.update(1)
    # Create the model
    pbar.set_description("Creating model")
    input_shape = (X_tr.shape[1], X_tr.shape[2])
    model = create_model(input_shape)
    pbar.update(1)
    # Train the model
    pbar.set_description("Training model")
    history = train_model(model, X_tr, y_tr)
    pbar.update(1)
    # Test the model
    pbar.set_description("Testing model")
    test_model(model, X_te, y_te)
    pbar.update(1)
    # Plot the training history
    pbar.set_description("Plotting history")
    plot_history(history)
    pbar.update(1)
    
    # Plotting test predictions
    pbar.set_description("Plotting predictions")
    plot_predictions(model, X_te, y_te, y_scaler)
    pbar.update(1)

    # Save the model
    pbar.set_description("Saving model")
    save_model(model)
    pbar.update(1)

    return 0
    
if __name__ =="__main__":
    main()