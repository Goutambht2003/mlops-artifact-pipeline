import json
import pytest
from sklearn.linear_model import LogisticRegression
from src.train import model

# Test 1: Configuration file loads correctly
def test_config_load():
    with open('config/config.json') as f:
        config = json.load(f)
    
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

# Test 2: Check model is a LogisticRegression instance
def test_model_type():
    assert isinstance(model, LogisticRegression)

# Test 3: Check if model is fitted
def test_model_fitted():
    assert hasattr(model, "coef_") and hasattr(model, "classes_")
