import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Load your dataset
data = pd.read_csv("winequality-red.csv", delimiter=';')

# Create a binary target variable: 1 if quality is >= 6, else 0
data['winequality'] = [1 if x >= 6 else 0 for x in data['quality']]
X = data.drop(['quality', 'winequality'], axis=1)
y = data['winequality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier
svm_classifier = SVC()

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Obtain the scores for both training and test sets
train_score_svm_classifier = svm_classifier.score(X_train_scaled, y_train)
test_score_svm_classifier = svm_classifier.score(X_test_scaled, y_test)

print('Training Score:', train_score_svm_classifier)
print('Test Score:', test_score_svm_classifier)

# Make predictions
svm_predictions = svm_classifier.predict(X_test_scaled)

# Classification report
print(classification_report(y_test, svm_predictions))

# Save the trained model as svm_model.joblib
joblib.dump(svm_classifier, 'svm_model.joblib')
