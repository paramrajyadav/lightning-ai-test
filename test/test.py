import numpy as np
from sklearn import model_selection as model_selection  # Importing from sci-kit learn
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Importing from sci-kit learn
from sklearn.linear_model import LogisticRegression  # Importing from sci-kit learn
from sklearn.metrics import accuracy_score, classification_report  # Importing from sci-kit learn

# Creating a synthetic dataset
np.random.seed(0)
X = np.random.randn(100, 2)  # 100 samples with 2 features
y = np.random.randint(0, 3, 100)  # Three classes (0, 1, 2)

# One-hot encoding y
encoder = OneHotEncoder(categories='auto', sparse=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

encoder1 = OneHotEncoder(categories='auto', sparse=True)
y_onehot1 = encoder1.fit_transform(y.reshape(-1, 1))


print(y_onehot)

print("**********************")

print(y_onehot1)
