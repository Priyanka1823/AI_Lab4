# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score


# Step 1: Load the UCI Thyroid Disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data"
column_names = ['T3_resin', 'Thyroxin', 'Triiodothyronine', 'Thyroidstimulating', 'TSH', 'class']
data = pd.read_csv(url, delim_whitespace=True, names=column_names)

# Step 2: Preprocess the data
# Initialize LabelEncoders for each column (assuming they are categorical)
encoders = {}
for col in data.columns:
    encoders[col] = LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])

    # Step 3: Split data into train and test sets
X = data.drop(columns=['class'])
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 4: Build a Bayesian Network structure (naive bayes structure for simplicity)
# Here, we'll assume all features directly depend on the class label
model = BayesianNetwork([('class', 'T3_resin'),
                         ('class', 'Thyroxin'),
                         ('class', 'Triiodothyronine'),
                         ('class', 'Thyroidstimulating'),
                         ('class', 'TSH')])

# Step 5: Train the Bayesian Network using Maximum Likelihood Estimation
train_data = pd.concat([X_train, y_train], axis=1)
model.fit(train_data, estimator=MaximumLikelihoodEstimator)
# Step 6: Perform inference using the trained model
inference = VariableElimination(model)

# Function to predict class based on feature values
def predict(instance):
    # Convert instance to dictionary
    evidence = instance.to_dict()
    # Ensure evidence values are integers (state numbers)
    evidence = {var: int(evidence[var]) for var in evidence}
    query_result = inference.map_query(variables=['class'], evidence=evidence)
    return query_result['class']

# Step 7: Evaluate the model
y_pred = X_test.apply(predict, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")