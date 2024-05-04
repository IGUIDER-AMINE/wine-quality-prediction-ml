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

# Define a function to remove outliers using the IQR method
def remove_outliers_iqr(df, columns):
    outlier_indices = set()  # Set to store indices of outliers
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Find indices of outliers and store them
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.update(outliers)
    # Drop rows containing outliers after the loop finishes
    df_cleaned = df.drop(outlier_indices)
    return df_cleaned
print("Origenal data : ",len(data))
# Remove outliers using the IQR method for selected columns
columns_to_check = ['alcohol','sulphates','citric acid','volatile acidity']
data = remove_outliers_iqr(data, columns_to_check)
print("Data without outliers: ",len(data))

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