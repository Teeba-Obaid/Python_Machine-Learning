# Group Classification with Random Forest

This repository contains a Python script that performs classification on grouped data using a Random Forest classifier. The script reads a dataset, processes group labels, and evaluates the model's performance on specific target variables.

## Prerequisites

- Python 3.x
- pandas
- scikit-learn

You can install the necessary Python packages using:
pip install pandas scikit-learn

# Usage
Place your dataset (data10.csv) in the same directory as the script.

Run the script:

bash
Copy code
python classification_script.py
The script will process the data, train a model, and print out the classification results for each group and target.

# Code Overview
**Load Data:** The script loads data from data10.csv.
**Assign Group Labels:** Groups are labeled as Control Group, Decision Table & Inductive Rules Group, and Inductive Rules Group.
**Encode Labels:** The group labels are encoded into numerical values.
**Classification Targets:** The script classifies the following targets:
- **Post-test:** Current 2 Valid links
- **Post-test:** Voltage Drop 2 Valid links
- **Rule Current:** Conservative Focusing
- **Rule Voltage Drop:** Conservative Focusing
- **Model Training:** A Random Forest classifier is used to train and predict the outcomes.
- **Results:** The script outputs accuracy and a detailed classification report for each group-target combination.

# Sample Output
The output includes accuracy and classification reports for each group and target.

Example:
### Group & Target: Control Group - Post-test: Current 2 Valid links

- **Accuracy:** 0.8000

#### Classification Report:

```plaintext
               precision    recall  f1-score   support

           1       0.83      1.00      0.91         5
           0       1.00      0.60      0.75         5

    accuracy                           0.80        10
   macro avg       0.92      0.80      0.83        10
weighted avg       0.92      0.80      0.83        10

