import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the CSV file into a DataFrame
df = pd.read_csv('data10.csv')

# Assign the correct group labels based on the row order
df['Group'] = ['Control Group'] * 20 + ['Decision Table & Inductive Rules Group'] * 20 + ['Inductive Rules Group'] * 20

# Encode the 'Group' column to numerical values
label_encoder = LabelEncoder()
df['Group'] = label_encoder.fit_transform(df['Group'])

# For binary and ordinal targets, use classification
classification_targets = [
    'Post-test: Current 2 Valid links', 'Post-test: Voltage Drop 2 Valid links',
    'Rule Current: Conservative Focusing', 'Rule Voltage Drop: Conservative Focusing'
]

# Function to transform 2s to 1s
def transform_targets(y):
    return y.replace(2, 1)

# Initialize a dictionary to store results
results = {}

# Loop through each group to train and evaluate a model
for group in df['Group'].unique():
    group_data = df[df['Group'] == group]  # Filter data for the current group
    for target in classification_targets:
        y = group_data[target]
        y_transformed = transform_targets(y)  # Transform target by replacing 2s with 1s
        X = group_data[['Group']]  # Feature is the encoded 'Group' column

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        # Store results
        results[f"{label_encoder.inverse_transform([group])[0]} - {target}"] = {
            'Accuracy': accuracy,
            'Classification Report': report
        }

# Display results
for key, metrics in results.items():
    print(f"Group & Target: {key}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Classification Report:\n{metrics['Classification Report']}")
