import numpy as np
from k29_algorithm import K29, binary_search


def test_binary_search():
    """Test the binary_search function."""
    print("Testing binary_search...")
    
    # Test 1: Simple linear function with root at 0.5
    def f1(x):
        return 2 * x - 1
    
    result = binary_search(f1)
    assert abs(result - 0.5) < 1e-2, f"Expected ~0.5, got {result}"
    print("  ✓ Test 1 passed: linear function")
    
    # Test 2: Function with root at 0.25
    def f2(x):
        return 4 * x - 1
    
    result = binary_search(f2)
    assert abs(result - 0.25) < 1e-2, f"Expected ~0.25, got {result}"
    print("  ✓ Test 2 passed: root at 0.25")
    
    # Test 3: Function always positive
    def f3(x):
        return x + 1
    
    result = binary_search(f3)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("  ✓ Test 3 passed: always positive function")
    
    print("All binary_search tests passed!\n")


def test_k29_simple():
    """Test K29 on simple synthetic data."""
    print("Testing K29 on simple synthetic data...")
    
    # Create simple 2D data: 1 continuous feature + 1 categorical feature
    np.random.seed(42)
    n_samples = 100
    
    # Continuous feature z
    z = np.random.randn(n_samples, 1)
    # Categorical feature g (0 or 1)
    g = np.random.randint(0, 2, size=(n_samples, 1))
    # Combine: X = [z, g]
    X = np.hstack([z, g])
    
    # Simple rule: y = 1 if z > 0, else 0
    y = (z.flatten() > 0).astype(float)
    
    # Split train/test
    train_size = 80
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Fit model
    model = K29(n_rff_features=50, gamma=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Check predictions are in [0, 1]
    assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions must be in [0, 1]"
    print("  ✓ Test 1 passed: predictions in valid range")
    
    # Check that predictions are reasonable (not all the same)
    assert len(np.unique(predictions)) > 1, "Predictions should vary"
    print("  ✓ Test 2 passed: predictions vary")
    
    print("All simple K29 tests passed!\n")


def test_k29_multiple_categories():
    """Test K29 with multiple categorical values."""
    print("Testing K29 with multiple categorical values...")
    
    np.random.seed(123)
    n_samples = 150
    
    # 2 continuous features + 1 categorical feature with 3 categories
    z = np.random.randn(n_samples, 2)
    g = np.random.randint(0, 3, size=(n_samples, 1))
    X = np.hstack([z, g])
    
    # Label based on first continuous feature and category
    y = ((z[:, 0] > 0) & (g.flatten() == 1)).astype(float)
    
    train_size = 100
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = K29(n_rff_features=50, gamma=1.0, random_state=123)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions must be in [0, 1]"
    print("  ✓ Test passed: multiple categories handled correctly")
    
    print("All multiple categories tests passed!\n")


def test_k29_single_prediction():
    """Test K29 with single sample prediction."""
    print("Testing K29 single sample prediction...")
    
    np.random.seed(456)
    n_samples = 50
    
    z = np.random.randn(n_samples, 1)
    g = np.random.randint(0, 2, size=(n_samples, 1))
    X = np.hstack([z, g])
    y = (z.flatten() > 0).astype(float)
    
    model = K29(n_rff_features=30, gamma=1.0, random_state=456)
    model.fit(X, y)
    
    # Single prediction
    x_new = np.array([0.5, 1])
    pred_single = model.predict(x_new)
    
    assert isinstance(pred_single, (float, np.floating)), "Single prediction should return scalar"
    assert 0 <= pred_single <= 1, "Prediction must be in [0, 1]"
    print("  ✓ Test passed: single prediction works")
    
    # Multiple predictions
    X_new = np.array([[0.5, 1], [-0.5, 0], [1.0, 1]])
    pred_multiple = model.predict(X_new)
    
    assert pred_multiple.shape == (3,), "Multiple predictions should return array"
    assert np.all((pred_multiple >= 0) & (pred_multiple <= 1)), "All predictions must be in [0, 1]"
    print("  ✓ Test passed: multiple predictions work")
    
    print("All single prediction tests passed!\n")


def test_k29_online_learning():
    """Test that K29 learns online (predictions change as more data is seen)."""
    print("Testing K29 online learning behavior...")
    
    np.random.seed(789)
    n_samples = 50
    
    z = np.random.randn(n_samples, 1)
    g = np.random.randint(0, 2, size=(n_samples, 1))
    X = np.hstack([z, g])
    y = (z.flatten() > 0).astype(float)
    
    # Test point
    x_test = np.array([0.0, 1])
    
    # Fit on increasing amounts of data
    predictions = []
    for i in range(10, n_samples, 10):
        model = K29(n_rff_features=30, gamma=1.0, random_state=789)
        model.fit(X[:i], y[:i])
        pred = model.predict(x_test)
        predictions.append(pred)
    
    # Predictions should be reasonable (not all identical or extreme)
    assert len(np.unique(np.round(predictions, 3))) > 1, "Predictions should change with more data"
    print("  ✓ Test passed: online learning behavior verified")
    
    print("All online learning tests passed!\n")


def test_k29_edge_cases():
    """Test edge cases."""
    print("Testing K29 edge cases...")
    
    np.random.seed(999)
    
    # Test 1: Very small dataset
    X_small = np.array([[0.5, 0], [-0.5, 1]])
    y_small = np.array([1.0, 0.0])
    
    model = K29(n_rff_features=10, gamma=1.0, random_state=999)
    model.fit(X_small, y_small)
    pred = model.predict(X_small[0])
    assert 0 <= pred <= 1, "Prediction must be valid"
    print("  ✓ Test 1 passed: very small dataset")
    
    # Test 2: All same category
    z = np.random.randn(20, 1)
    g = np.zeros((20, 1))  # All same category
    X = np.hstack([z, g])
    y = (z.flatten() > 0).astype(float)
    
    model = K29(n_rff_features=20, gamma=1.0, random_state=999)
    model.fit(X, y)
    pred = model.predict(X[0])
    assert 0 <= pred <= 1, "Prediction must be valid"
    print("  ✓ Test 2 passed: all same category")
    
    # Test 3: High-dimensional continuous features
    z = np.random.randn(30, 5)  # 5 continuous features
    g = np.random.randint(0, 2, size=(30, 1))
    X = np.hstack([z, g])
    y = (z[:, 0] > 0).astype(float)
    
    model = K29(n_rff_features=50, gamma=1.0, random_state=999)
    model.fit(X, y)
    pred = model.predict(X[0])
    assert 0 <= pred <= 1, "Prediction must be valid"
    print("  ✓ Test 3 passed: high-dimensional features")
    
    print("All edge case tests passed!\n")


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("Running K29 Algorithm Tests")
    print("=" * 60)
    print()
    
    try:
        test_binary_search()
        test_k29_simple()
        test_k29_multiple_categories()
        test_k29_single_prediction()
        test_k29_online_learning()
        test_k29_edge_cases()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
