import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load stored responses
data = pd.read_csv('user_responses.csv')

# Drop non-predictive columns (adjust as needed)
drop_columns = ['Q1', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8', 'Q9', 'Timestamp']
X = data.drop(columns=drop_columns + ['C'])
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

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# Save model and encoders
joblib.dump(model, 'prakriti_model_new.pkl')
joblib.dump(label_encoders, 'label_encoders_new.pkl')
joblib.dump(le_y, 'target_encoder_new.pkl')