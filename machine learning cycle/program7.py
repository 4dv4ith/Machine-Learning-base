import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Sample synthetic dataset (for demonstration purposes)
data = {
    'Age': np.random.randint(20, 60, 1000),
    'Gender': np.random.choice(['Male', 'Female'], 1000),
    'Education': np.random.choice(['Diploma', 'Degree', 'Postgraduate'], 1000),
    'Industry': np.random.choice(['Manufacturing', 'IT', 'Finance', 'Healthcare'], 1000),
    'Residence': np.random.choice(['Metro', 'Non-Metro'], 1000),
    'Income': np.random.choice(['Below 4L', '4L-15L', 'Above 15L'], 1000)
}

# Create DataFrame
df = pd.DataFrame(data)

# Custom LabelEncoder to handle unseen labels
class SafeLabelEncoder(LabelEncoder):
    def transform(self, values):
        """Transform values, assigning -1 for unseen labels."""
        return [self.classes_.tolist().index(val) if val in self.classes_ else -1 for val in values]

# Train label encoders properly
gender_encoder = SafeLabelEncoder()
gender_encoder.fit(df['Gender'])

education_encoder = SafeLabelEncoder()
education_encoder.fit(df['Education'])

industry_encoder = SafeLabelEncoder()
industry_encoder.fit(df['Industry'])

residence_encoder = SafeLabelEncoder()
residence_encoder.fit(df['Residence'])

# Encode categorical variables
for col in ['Gender', 'Education', 'Industry', 'Residence', 'Income']:
    df[col] = gender_encoder.fit_transform(df[col])

# Define features and target
X = df.drop(columns=['Income'])
y = df['Income']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Define the individual for prediction
individual = pd.DataFrame({
    'Age': [35],
    'Gender': gender_encoder.transform(['Male'])[0],
    'Education': education_encoder.transform(['Diploma'])[0],
    'Industry': industry_encoder.transform(['Manufacturing'])[0],
    'Residence': residence_encoder.transform(['Metro'])[0]
})

# Make prediction
prediction = rf_model.predict(individual)
predicted_income = gender_encoder.inverse_transform(prediction)

print("Predicted Income Category:", predicted_income[0])