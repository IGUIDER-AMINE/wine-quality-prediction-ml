import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import joblib

# Load your dataset
data = pd.read_csv("winequality.csv", delimiter=';')

print(data.head())
# =====
# Select the specified columns
selected_cols = ['alcohol', 'sulphates', 'citric acid', 'volatile acidity']
data_selected = data[selected_cols]

# Initialize a variable to store the indices of outliers
outlier_indices = []

# Loop through each selected column and identify outliers
for col in selected_cols:
    # Calculate mean and standard deviation for the current column
    mean_col = data_selected[col].mean()
    std_col = data_selected[col].std()

    # Define a threshold (e.g., 3 times the standard deviation)
    threshold = 3 * std_col

    # Identify outliers for the current column and store their indices
    col_outliers = data_selected[(data_selected[col] < mean_col - threshold) | (data_selected[col] > mean_col + threshold)].index
    outlier_indices.extend(col_outliers)

# Remove duplicate indices (if any)
outlier_indices = list(set(outlier_indices))

# Drop rows containing outliers
data_no_outliers = data.drop(outlier_indices)

# Calculate the number of rows removed
num_rows_removed = len(outlier_indices)

# Display the number of rows removed
print("Number of rows removed:", num_rows_removed)

# Print data_no_outliers and data with counts
print("\nCount of data_no_outliers:", len(data_no_outliers))
print("\nCount of original data:", len(data))
data = data_no_outliers
print("\ndata len:", len(data))
# Create a binary target variable: 1 if quality is >= 6, else 0
data['winequality'] = [1 if x >= 6 else 0 for x in data['quality']]
X = data.loc[:,['alcohol','sulphates','citric acid','volatile acidity']]
y = data['winequality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.head())

# Feature scaling
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# Transform the testing data
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# Create an SVM classifier with probability estimation enabled
svm_classifier = SVC(probability=True)
# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Create a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier()
# Train the classifier
gb_classifier.fit(X_train_scaled, y_train)


# Voting ensemble
voting_model = VotingClassifier(estimators=[
    ('Gradient Boosting', gb_classifier),
    ('SVM', svm_classifier)
], voting='soft')
voting_model.fit(X_train_scaled, y_train)

# Evaluate the models
models = [gb_classifier, svm_classifier, voting_model]
for i, model in enumerate(models, start=2):
    train_score = voting_model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Model {i} - Training Score: {train_score}, Test Score: {test_score}")

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Classification report
    print(f"Classification Report for Model {i}:")
    print(classification_report(y_test, y_pred))

# Save the trained model as gb_model.joblib
joblib.dump(voting_model, 'voting_model.joblib')
# Save the scaler as scaler.joblib
joblib.dump(scaler, 'scaler.joblib')


# Given input values
volatile_acidity = 0.35
citric_acid = 0.46
sulphates = 0.86
alcohol = 12.8

# Load the scaler from file
scaler = joblib.load('scaler.joblib')
model = joblib.load('voting_model.joblib')

# Scale the input features
input_df = pd.DataFrame([[alcohol,sulphates, citric_acid,volatile_acidity]], columns=['alcohol','sulphates','citric acid','volatile acidity'])
input_features = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_features)
print(input_features)
print(prediction)
# Display the prediction
if prediction[0] == 1:
    print('Predicted wine quality: Good')
else:
    print('Predicted wine quality: Bad')