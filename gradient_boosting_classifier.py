import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib

# Load your dataset
data = pd.read_csv("winequality-red.csv", delimiter=';')

print(data.head())
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

# Save the scaler as scaler.joblib
joblib.dump(scaler, 'scaler.joblib')

# Create a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier()

# Train the classifier
gb_classifier.fit(X_train_scaled, y_train)

# Obtain the scores for both training and test sets
train_score_gb_classifier = gb_classifier.score(X_train_scaled, y_train)
test_score_gb_classifier = gb_classifier.score(X_test_scaled, y_test)

print('Training Score:', train_score_gb_classifier)
print('Test Score:', test_score_gb_classifier)

# Make predictions
gb_predictions = gb_classifier.predict(X_test_scaled)

# Classification report
print(classification_report(y_test, gb_predictions))

# Save the trained model as gb_model.joblib
joblib.dump(gb_classifier, 'gb_model.joblib')

# Given input values
fixed_acidity = 5.9
volatile_acidity = 0.55
citric_acid = 0.1
residual_sugar = 2.2
chlorides = 0.062
free_sulfur_dioxide = 39.0
total_sulfur_dioxide = 51.0
density = 0.99512
pH = 3.52
sulphates = 0.76
alcohol = 11.2

# Load the scaler from file
scaler = joblib.load('scaler.joblib')
model = joblib.load('gb_model.joblib')

# Scale the input features
input_features = scaler.transform([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                     free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])

# Make prediction
prediction = model.predict(input_features)
print(input_features)
print(prediction)
# Display the prediction
if prediction[0] == 1:
    print('Predicted wine quality: Good')
else:
    print('Predicted wine quality: Bad')