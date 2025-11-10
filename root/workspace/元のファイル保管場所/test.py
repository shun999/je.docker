import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# --- 1. Transformer Block Implementation ---
# This is the core component of the Transformer model.
class TransformerBlock(layers.Layer):
    """
    A single Transformer block consisting of multi-head self-attention
    and a feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        # Multi-Head Self-Attention layer
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Feed-Forward Network
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        # Dropout
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Multi-head attention sublayer
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) # Residual connection
        
        # Feed-forward network sublayer
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output) # Residual connection

# --- 2. Model Creation Function ---
def create_tabular_transformer(num_features, num_classes, embed_dim=32, num_heads=4, ff_dim=32):
    """
    Creates the full Transformer model for tabular data classification.
    """
    inputs = layers.Input(shape=(num_features,))
    
    # 1. Feature Embedding
    # Project the input features into a higher dimensional space (embedding)
    x = layers.Dense(embed_dim)(inputs)
    
    # 2. Reshape for Transformer
    # Transformer expects a sequence: (batch_size, sequence_length, embed_dim)
    # For tabular data, we can treat the features as a sequence of length 1.
    x = layers.Reshape((1, embed_dim))(x)
    
    # 3. Apply Transformer Block
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    
    # 4. Post-processing for Classification
    x = layers.GlobalAveragePooling1D()(x) # Pool the sequence
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    
    # 5. Output Layer
    # Use 'softmax' for multiclass classification
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

# --- 3. Data Preparation ---
print("Generating synthetic data...")
# Generate a synthetic dataset for 5-class classification
# X, y = make_classification(
#     n_samples=5000,
#     n_features=20,
#     n_informative=10,
#     n_redundant=5,
#     n_classes=5,
#     n_clusters_per_class=2,
#     random_state=42
# )

df = pd.read_excel("/root/workspace/深層学習.xlsx")
X = df[['left_x', 'left_y', 'right_x', 'right_y', 'left_pupil', 'right_pupil']]
y = df["is_correct"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode the labels for categorical_crossentropy
y = keras.utils.to_categorical(y, num_classes=5)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print("-" * 30)

# --- 4. Model Compilation and Training ---
num_features = X_train.shape[1]
num_classes = y_train.shape[1]

# Create the model instance
model = create_tabular_transformer(num_features, num_classes)

# Compile the model
# Use 'categorical_crossentropy' for multiclass classification
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Print model summary
model.summary()

# Train the model
print("\nStarting model training...")
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=20, # Increased epochs for better convergence
    validation_split=0.2,
    verbose=1
)
print("Model training finished.")
print("-" * 30)


# --- 5. Evaluation ---
print("Evaluating model on test data...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")