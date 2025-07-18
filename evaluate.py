from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load trained model
model = joblib.load("model.pkl")

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Performance Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')  # multi-class
loss = log_loss(y_test, y_prob)

print(f"📊 Performance Report:")
print(f"✅ Accuracy: {acc:.2f}")
print(f"✅ F1 Score: {f1:.2f}")
print(f"✅ Cross-Entropy Loss: {loss:.4f}")
