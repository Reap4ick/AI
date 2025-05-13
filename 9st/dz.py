import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

def load_train_data():
    """Loads the Titanic training dataset."""
    return pd.read_csv('workspace/csv_files/train.csv')

def load_test_data():
    """Loads the Titanic test dataset."""
    return pd.read_csv('workspace/csv_files/test.csv')

def preprocess_data(df, scaler=None, training_columns=None, is_train=True):
    """Preprocesses the data: fills missing values, encodes categorical variables, scales numerical features."""
    passenger_info = df.copy()
    
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    df_for_model = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
    df_for_model = pd.get_dummies(df_for_model, columns=['Sex', 'Embarked'], drop_first=True)
    
    if is_train:
        X = df_for_model.drop(columns=['Survived'])
        y = df_for_model['Survived']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler, X.columns, passenger_info
    else:
        for col in training_columns:
            if col not in df_for_model.columns:
                df_for_model[col] = 0
        df_for_model = df_for_model[training_columns]
        X_scaled = scaler.transform(df_for_model)
        return X_scaled, passenger_info

def build_model(input_shape):
    """Builds and compiles a neural network for binary classification."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Trains the model and returns the training history."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    return history

if __name__ == "__main__":
    train_df = load_train_data()
    X_train, y_train, scaler, training_columns, _ = preprocess_data(train_df, is_train=True)

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = build_model(X_train.shape[1])
    history = train_model(model, X_train_split, y_train_split, X_val, y_val)

    test_df = load_test_data()
    X_test, passenger_info = preprocess_data(test_df, scaler, training_columns, is_train=False)

    y_pred = model.predict(X_test).flatten()
    test_df['Survived'] = (y_pred > 0.5).astype(int)

    test_df.to_csv('titanic_test_with_predictions.csv', index=False)
