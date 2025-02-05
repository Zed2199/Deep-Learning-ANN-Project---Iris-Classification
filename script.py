import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


def load_and_prepare_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    print("Dataset Shape:", X.shape)
    print("\nFeature Statistics:")
    print(X.describe())

    plt.figure(figsize=(10, 6))
    X.boxplot()
    plt.title('Feature Distributions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.show()

    return X, y


def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim,
              kernel_initializer='he_normal'),
        BatchNormalization(),

        Dense(32, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        BatchNormalization(),

        Dense(16, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.2),
        BatchNormalization(),

        Dense(3, activation='softmax')  # 3 classes for Iris
    ])

    return model


def train_and_evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return history


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    print("1. Loading and exploring data...")
    X, y = load_and_prepare_data()

    print("\n2. Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)

    print("\n3. Creating model...")
    model = create_model(X_train.shape[1])
    model.summary()

    print("\n4. Training model...")
    history = train_and_evaluate_model(
        model, X_train, X_val, X_test, y_train, y_val, y_test
    )

    print("\n5. Plotting results...")
    plot_training_history(history)


if __name__ == "__main__":
    main()