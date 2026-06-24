import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os

def train_alphanumeric_model():
    print("==================================================")
    # 1. Load MNIST Digits (0-9)
    print("🔄 Step 1: Loading MNIST Digits Dataset...")
    (X_digits_train, y_digits_train), (X_digits_test, y_digits_test) = tf.keras.datasets.mnist.load_data()
    
    # Combine MNIST train and test sets to maximize training volume
    X_digits = np.concatenate((X_digits_train, X_digits_test), axis=0)
    y_digits = np.concatenate((y_digits_train, y_digits_test), axis=0)
    print(f"✅ Loaded {len(X_digits)} digit samples (Classes 0-9).")

    # 2. Load A-Z CSV Alphabet Dataset (A-Z mapped to 10-35)
    csv_path = "A_Z Handwritten Data.csv"
    print(f"\n🔄 Step 2: Loading {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: '{csv_path}' not found in this folder!")
        print("Please ensure your Kaggle A-Z CSV file is placed in the same directory.")
        return

    # Read CSV matrix efficiently
    az_data = pd.read_csv(csv_path).values
    X_alphas = az_data[:, 1:].reshape(-1, 28, 28)
    
    # CRITICAL: Shift labels by 10 so they don't overwrite digit classes
    y_alphas = az_data[:, 0] + 10  
    print(f"✅ Loaded {len(X_alphas)} alphabet samples (Shifted to Classes 10-35).")

    # 3. Concatenate and Preprocess Datasets
    print("\n🔄 Step 3: Merging arrays into Unified Alphanumeric Matrix...")
    X_combined = np.concatenate((X_digits, X_alphas), axis=0)
    y_combined = np.concatenate((y_digits, y_alphas), axis=0)

    # Normalize pixel matrices from [0, 255] to [0.0, 1.0]
    X_combined = X_combined.astype("float32") / 255.0
    
    # Expand dimensions to match CNN Input Layer expectations: (Batch, 28, 28, 1 Channel)
    X_combined = np.expand_dims(X_combined, -1)

    # Stratified-style train/test split for training stability
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42
    )
    print(f"📊 Total Combined Dataset Size: {len(X_combined)} entries.")
    print(f"   ↳ Training Matrix Shape:   {X_train.shape}")
    print(f"   ↳ Testing Matrix Shape:    {X_test.shape}")

    # 4. Construct the Upgraded Deep Learning Architecture
    print("\n🧠 Step 4: Constructing Alphanumeric Convolutional Architecture...")
    model = models.Sequential([
        # Convolution Layer 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Convolution Layer 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense Layer Processing
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Prevents overfitting during the matrix merge
        
        # Output Classification Layer (36 classes: 0-9 digits + 26 letters)
        layers.Dense(36, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 5. Execute Training Engine
    print("\n🚀 Step 5: Launching Neural Network Training Sequence...")
    # 4-5 epochs is ideal; anything more might overfit the simple digit structures
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=4,
        batch_size=128,
        verbose=1
    )

    # 6. Save the Output Tensor Model
    output_model_name = "alphanumeric_model.h5"
    print(f"\n💾 Step 6: Saving trained structures to disk...")
    model.save(output_model_name)
    print(f"🎉 SUCCESS! '{output_model_name}' has been created and is ready for app.py deployment.")
    print("==================================================")

if __name__ == "__main__":
    train_alphanumeric_model()