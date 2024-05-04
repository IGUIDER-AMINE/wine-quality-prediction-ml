import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report

# Load data
path = "kddcup.data_10_percent_corrected"
cols = ["logged_in", "count", "dst_host_count", "protocol_type", "srv_count", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "hot", "root_shell", "target"]
df = pd.read_csv(path, names=cols)

# Map target classes to ['normal', 'u2r', 'dos', 'r2l', 'probe']
attacks_types = {
    'normal': 'normal',
    'back': 'dos',
    'buffer_overflow': 'u2r',
    'ftp_write': 'r2l',
    'guess_passwd': 'r2l',
    'imap': 'r2l',
    'ipsweep': 'probe',
    'land': 'dos',
    'loadmodule': 'u2r',
    'multihop': 'r2l',
    'neptune': 'dos',
    'nmap': 'probe',
    'perl': 'u2r',
    'phf': 'r2l',
    'pod': 'dos',
    'portsweep': 'probe',
    'rootkit': 'u2r',
    'satan': 'probe',
    'smurf': 'dos',
    'spy': 'r2l',
    'teardrop': 'dos',
    'warezclient': 'r2l',
    'warezmaster': 'r2l',
}

# Adding Attack Type column
df['Attack Type'] = df.target.apply(lambda r: attacks_types[r[:-1]])

# Drop the 'target' column
df.drop(columns=['target'], inplace=True)

# Define features and target
X = df.drop(columns=['Attack Type'])
y = df['Attack Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Decision Tree model
model2 = DecisionTreeClassifier(criterion="entropy", max_depth=4)
model2.fit(X_train_scaled, y_train)

# Random Forest model
model3 = RandomForestClassifier(n_estimators=30)
model3.fit(X_train_scaled, y_train)

# Voting ensemble
ensemble_model = VotingClassifier(estimators=[
    ('Random Forest', model3),
    ('Decision Tree', model2)
], voting='soft')
ensemble_model.fit(X_train_scaled, y_train)

#################
# Evaluate the models
models = [model2, model3, ensemble_model]
for i, model in enumerate(models, start=2):
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Model {i} - Training Score: {train_score}, Test Score: {test_score}")

    y_pred = model.predict(X_test_scaled)
    print(f"Classification Report for Model {i}:")
    print(classification_report(y_test, y_pred))

# Save the models and scaler
joblib.dump(scaler, 'minmax_scaler.joblib')
joblib.dump(ensemble_model, 'voting_ensemble_model.joblib')

# Input values for prediction
input_values = {
    'logged_in': 1,
    'count': 6,
    'dst_host_count': 5,
    'protocol_type': 1,  # Assuming 1 represents 'tcp' based on previous code
    'srv_count': 5,
    'dst_host_diff_srv_rate': 0.0,
    'dst_host_same_src_port_rate': 0.2,
    'dst_host_srv_diff_host_rate': 0.33,
    'hot': 0,
    'root_shell': 0
}

# Convert input values to a DataFrame
input_df = pd.DataFrame([input_values])

# Scale the input features using the same scaler used for training
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

# Make predictions using the ensemble model
prediction = ensemble_model.predict(input_scaled)

# Print the prediction
print("Predicted Attack Type:", prediction[0])


