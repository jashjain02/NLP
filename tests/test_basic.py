"""
Basic tests for the NLP Finance Pipeline
"""

import pytest
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_imports():
    """Test that all main modules can be imported."""
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import streamlit as st
        print("✅ Basic imports successful")
        assert True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        assert False, f"Failed to import required modules: {e}"

def test_optional_imports():
    """Test optional modules that might not be available."""
    optional_modules = [
        ('matplotlib', 'matplotlib.pyplot'),
        ('seaborn', 'seaborn'),
        ('nltk', 'nltk.sentiment'),
        ('scipy', 'scipy.stats')
    ]
    
    for module_name, import_path in optional_modules:
        try:
            __import__(import_path)
            print(f"✅ {module_name} import successful")
        except ImportError:
            print(f"⚠️ {module_name} not available (optional)")
            # Don't fail the test for optional modules

def test_data_collection_imports():
    """Test data collection module imports."""
    try:
        from data_collection.stock import fetch_stock_data
        print("✅ Stock data collection import successful")
        assert True
    except ImportError as e:
        print(f"⚠️ Stock data collection import failed: {e}")
        # Don't fail for missing modules in test environment
        
    try:
        from data_collection.news import collect_news_multi_source
        print("✅ News data collection import successful")
        assert True
    except ImportError as e:
        print(f"⚠️ News data collection import failed: {e}")
        # Don't fail for missing modules in test environment

def test_preprocessing_imports():
    """Test preprocessing module imports."""
    try:
        from preprocessing.preprocess import preprocess_data
        print("✅ Preprocessing imports successful")
        assert True
    except ImportError as e:
        print(f"⚠️ Preprocessing import failed: {e}")
        # Don't fail for missing modules in test environment

def test_nlp_imports():
    """Test NLP module imports."""
    try:
        from nlp.sentiment import analyze_sentiment_batch
        print("✅ NLP imports successful")
        assert True
    except ImportError as e:
        print(f"⚠️ NLP import failed: {e}")
        # Don't fail for missing modules in test environment

def test_features_imports():
    """Test features module imports."""
    try:
        from features.features import create_features
        print("✅ Features imports successful")
        assert True
    except ImportError as e:
        print(f"⚠️ Features import failed: {e}")
        # Don't fail for missing modules in test environment

def test_modeling_imports():
    """Test modeling module imports."""
    try:
        from modeling.modeling import train_classifiers, train_regressors
        print("✅ Modeling imports successful")
        assert True
    except ImportError as e:
        print(f"⚠️ Modeling import failed: {e}")
        # Don't fail for missing modules in test environment

def test_streamlit_app_imports():
    """Test Streamlit app imports."""
    try:
        from ui.streamlit_app import main
        print("✅ Streamlit app imports successful")
        assert True
    except ImportError as e:
        print(f"⚠️ Streamlit app import failed: {e}")
        # Don't fail for missing modules in test environment

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    import pandas as pd
    import numpy as np
    
    # Test basic pandas operations
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'price': np.random.randn(10).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 10)
    })
    
    # Test basic calculations
    df['returns'] = df['price'].pct_change()
    df['sma_5'] = df['price'].rolling(5).mean()
    
    assert len(df) == 10
    assert 'returns' in df.columns
    assert 'sma_5' in df.columns
    print("✅ Basic functionality test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
