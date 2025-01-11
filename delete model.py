import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
from joblib import dump

warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_excel('data/chronic_diseases_recommendations_expanded copy.xlsx')

# Data preprocessing
print("Preprocessing data...")

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Chronic Disease', 'gender', 'severity_of_pain']
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Encode age with more granular mapping
age_mapping = {
    '0-10': 5, '10-20': 15, '20-30': 25, '30-40': 35,
    '40-50': 45, '50-60': 55, '60-70': 65, '70-80': 75,
    '80-90': 85, '90-100': 95
}
df['age'] = df['age'].map(age_mapping).fillna(df['age'].mode()[0])

# Feature engineering
print("Performing feature engineering...")

# Create interaction features
df['age_severity'] = df['age'] * df['severity_of_pain']
df['disease_severity'] = df['Chronic Disease'] * df['severity_of_pain']

# Define input features
feature_columns = [
    'age', 'gender', 'severity_of_pain', 'Chronic Disease',
    'age_severity', 'disease_severity'
]

X = df[feature_columns]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize storage
models = {}
encoders = {}
accuracies = {}

# Define target fields
targets = [
    'dietmorning', 'dietafternoon', 'dietnight',
    'Yoga Recommendation (Pose 1)', 'Yoga Recommendation (Pose 2)',
    'Yoga Recommendation (Pose 3)', 'Yoga Recommendation (Pose 4)'
]

print("\nTraining models for each target...")

for target in targets:
    print(f"\nProcessing: {target}")
    
    # Encode target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target].astype(str))
    encoders[target] = le_target
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Apply SMOTE for balanced classes
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Random Forest with optimized parameters
    rf_params = {
        'n_estimators': [200, 300],
        'max_depth': [15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        rf_params,
        cv=StratifiedKFold(n_splits=5),
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Train model
    print(f"Training model for {target}...")
    rf.fit(X_train_res, y_train_res)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[target] = accuracy
    
    print(f"Best parameters: {rf.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    
    # Store the best model
    models[target] = rf.best_estimator_
    
    # Feature importance analysis
    importance = rf.best_estimator_.feature_importances_
    feature_imp = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_imp)

# Display final accuracies
print("\nFinal Accuracies:")
for target, accuracy in accuracies.items():
    print(f"{target}: {accuracy:.4f}")

# Save models and encoders
print("\nSaving models and encoders...")
dump({
    'models': models,
    'scaler': scaler,
    'encoders': encoders,
    'label_encoders': label_encoders
}, 'models_diet_yoga.joblib')

print("\nTraining completed successfully!")