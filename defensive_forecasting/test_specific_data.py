"""
Test K29 algorithm on specific dataset:
- x_1, x_2 are i.i.d uniform on [0,1]
- g is the indicator that x_1 > 0.5
- If g=1 then y=x_2, else y=x_1
"""

import numpy as np
from k29_algorithm import K29

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_data(n_samples=100, random_state=42):
    """
    Generate the test dataset.
    
    Parameters:
    n_samples (int): Number of samples to generate
    random_state (int): Random seed
    
    Returns:
    X (array): Features of shape (n_samples, 3) where [x_1, x_2, g]
    y (array): Labels of shape (n_samples,)
    """
    rng = np.random.RandomState(random_state)
    
    # Generate x_1 and x_2 as i.i.d uniform on [0,1]
    x_1 = rng.uniform(0, 1, size=n_samples)
    x_2 = rng.uniform(0, 1, size=n_samples)
    
    # g is the indicator that x_1 > 0.5
    g = (x_1 > 0.5).astype(int)
    
    # y = x_2 if g=1, else y = x_1
    y = np.where(g == 1, x_2, x_1)
    
    # Combine into X: [x_1, x_2, g]
    X = np.column_stack([x_1, x_2, g])
    
    return X, y


def main():
    print("=" * 60)
    print("Testing K29 on Specific Dataset")
    print("=" * 60)
    print()
    
    # Generate data
    print("Generating dataset...")
    X, y = generate_data(n_samples=100, random_state=42)
    
    print(f"  Dataset size: {len(X)}")
    print(f"  Feature shape: {X.shape}")
    print(f"  x_1 range: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
    print(f"  x_2 range: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")
    print(f"  g values: {np.unique(X[:, 2])}")
    print(f"  g=1 ratio: {(X[:, 2] == 1).mean():.3f}")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  y mean: {y.mean():.3f}")
    print()
    
    # Split into train and test (80/20)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print()
    
    # Fit the model
    print("Fitting K29 model...")
    model = K29(
        n_rff_features=100,
        gamma=1.0,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("  ✓ Model fitted successfully")
    print()
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  Mean prediction: {predictions.mean():.3f}")
    print(f"  Mean true label: {y_test.mean():.3f}")
    print()
    
    # Evaluate
    print("Evaluation:")
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))
    
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print()
    
    # Analyze by category
    print("Analysis by category:")
    for g_val in [0, 1]:
        mask = X_test[:, 2] == g_val
        if mask.sum() > 0:
            pred_g = predictions[mask]
            true_g = y_test[mask]
            mse_g = np.mean((pred_g - true_g) ** 2)
            mae_g = np.mean(np.abs(pred_g - true_g))
            print(f"  g={g_val}:")
            print(f"    Samples: {mask.sum()}")
            print(f"    MSE: {mse_g:.4f}")
            print(f"    MAE: {mae_g:.4f}")
            print(f"    Mean prediction: {pred_g.mean():.3f}")
            print(f"    Mean true label: {true_g.mean():.3f}")
    print()
    
    # Show some example predictions
    print("Example predictions (first 10 test samples):")
    print("  x_1    x_2    g    y_true    y_pred    error")
    print("  " + "-" * 50)
    for i in range(min(10, len(X_test))):
        x1, x2, g = X_test[i]
        y_true = y_test[i]
        y_pred = predictions[i]
        error = abs(y_pred - y_true)
        print(f"  {x1:.3f}  {x2:.3f}  {int(g)}    {y_true:.3f}     {y_pred:.3f}    {error:.3f}")
    print()
    
    # Plot predictions vs true values
    if HAS_MATPLOTLIB:
        try:
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.scatter(y_test, predictions, alpha=0.6)
            plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
            plt.xlabel('True y')
            plt.ylabel('Predicted y')
            plt.title('Predictions vs True Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            errors = predictions - y_test
            plt.hist(errors, bins=20, edgecolor='black')
            plt.xlabel('Prediction Error (pred - true)')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('k29_test_results.png', dpi=150, bbox_inches='tight')
            print("  ✓ Saved plot to 'k29_test_results.png'")
            print()
        except Exception as e:
            print(f"  Could not create plot: {e}")
            print()
    else:
        print("  (Skipping plot - matplotlib not available)")
        print()
    
    print("=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
