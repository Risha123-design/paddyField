import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('data/fertilizer_data.csv')
df.columns = df.columns.str.strip().str.lower()

# Encode categorical features
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()
le_soil = LabelEncoder()

df['crop'] = le_crop.fit_transform(df['crop'])
df['fertilizer'] = le_fertilizer.fit_transform(df['fertilizer'])
df['soil'] = le_soil.fit_transform(df['soil'])

# Features and Target
X = df[['temperature', 'moisture', 'rainfall', 'ph', 'nitrogen',
        'phosphorous', 'potassium', 'carbon', 'soil', 'crop']]
y_class = df['fertilizer']

# Split dataset
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train_class)

# Save model and encoders
with open('models/fertilizer_clf.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('models/le_crop.pkl', 'wb') as f:
    pickle.dump(le_crop, f)

with open('models/le_fertilizer.pkl', 'wb') as f:
    pickle.dump(le_fertilizer, f)

with open('models/le_soil.pkl', 'wb') as f:
    pickle.dump(le_soil, f)

print("Fertilizer classification model trained and saved successfully.")
