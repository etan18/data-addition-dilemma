"""
Example usage of the K29 algorithm.

This script demonstrates how to use the K29 classifier with Random Fourier Features
and categorical features.
"""

import numpy as np
from k29_algorithm import K29


def main():
    print("=" * 60)
    print("K29 Algorithm Example Usage")
    print("=" * 60)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    n_samples = 200
    n_continuous = 2  # Number of continuous features
    n_categories = 3  # Number of possible categorical values
    
    # Continuous features z (d-1 dimensions)
    z = np.random.randn(n_samples, n_continuous)
    
    # Categorical feature g (last dimension, values 0, 1, or 2)
    g = np.random.randint(0, n_categories, size=(n_samples, 1))
    
    # Combine: X = [z, g] where z has d-1 columns and g has 1 column
    X = np.hstack([z, g])
    
    # Create labels: y = 1 if first continuous feature > 0 AND category == 1, else 0
    y = ((z[:, 0] > 0) & (g.flatten() == 1)).astype(float)
    
    print(f"  Data shape: {X.shape}")
    print(f"  Continuous features: {n_continuous}")
    print(f"  Categorical values: {n_categories}")
    print(f"  Positive class ratio: {y.mean():.2f}")
    print()
    
    # Split into train and test
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print()
    
    # Initialize and fit the model
    print("Fitting K29 model...")
    model = K29(
        n_rff_features=100,  # Number of Random Fourier Features
        gamma=1.0,           # RBF kernel parameter
        random_state=42      # For reproducibility
    )
    
    model.fit(X_train, y_train)
    print("  ✓ Model fitted successfully")
    print()
    
    # Make predictions on test set
    print("Making predictions on test set...")
    predictions = model.predict(X_test)
    print(f"  ✓ Predictions shape: {predictions.shape}")
    print(f"  ✓ Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  ✓ Mean prediction: {predictions.mean():.3f}")
    print()
    
    # Evaluate predictions
    print("Evaluation:")
    # Convert probabilities to binary predictions (threshold at 0.5)
    binary_preds = (predictions >= 0.5).astype(float)
    accuracy = (binary_preds == y_test).mean()
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  True positive rate: {y_test.mean():.3f}")
    print(f"  Predicted positive rate: {binary_preds.mean():.3f}")
    print()
    
    # Example: Single prediction
    print("Example: Single prediction")
    x_new = np.array([1.5, -0.3, 1])  # [z1, z2, g]
    pred = model.predict(x_new)
    print(f"  Input: continuous=[{x_new[0]:.1f}, {x_new[1]:.1f}], categorical={int(x_new[2])}")
    print(f"  Prediction: {pred:.3f}")
    print()
    
    # Show some training history statistics
    print("Training history statistics:")
    print(f"  Number of training samples: {len(model.history_y)}")
    print(f"  Mean training prediction: {np.mean(model.history_p):.3f}")
    print(f"  Mean training label: {np.mean(model.history_y):.3f}")
    print()
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
