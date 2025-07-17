import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

# Load config from JSON
with open('config/config.json') as f:
    config = json.load(f)

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Create and train Logistic Regression model
model = LogisticRegression(
    C=config["C"],
    solver=config["solver"],
    max_iter=config["max_iter"]
)
model.fit(X, y)

# Save the model
joblib.dump(model, 'model_train.pkl')

print("✅ Model trained and saved as model_train.pkl")
