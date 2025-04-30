import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('fold_all_c.csv')

# Define columns to drop
drop_columns = ['SampleID', 'C', 'Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5', 'Fold6', 'Fold7', 'Fold8', 'Fold9', 'Fold10']
non_predictive = ['First Name', 'Gender', 'Permanent Address', 'Phone Number', 'Email Address', 
                  'Code No', 'Age', 'Date', 'Interviewer Name', 'Genetic Analysis', 
                  'Weight', 'Height', 'F1', 'F2', 'GENDER', 'AGE', 'GENETIC']
drop_columns.extend([col for col in non_predictive if col in data.columns])

# Separate features and target
X = data.drop(columns=drop_columns)
y = data['C']

# Encode categorical features
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le

# Encode target
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# Save model and encoders
joblib.dump(model, 'prakriti_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(le_y, 'target_encoder.pkl')

# Example prediction
sample = X_test.iloc[0:1]
probs = model.predict_proba(sample)[0]
dosha_names = le_y.inverse_transform([0, 1, 2])
print("Sample Predicted Probabilities:")
for dosha, prob in zip(dosha_names, probs):
    print(f"{dosha}: {prob*100:.2f}%")