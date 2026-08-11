import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'telecom_churn.csv')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Drop columns that are not useful for the model
    # We will drop customer_id and date_of_registration (since it's a date string)
    df = df.drop(columns=['customer_id', 'date_of_registration'])
    
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    # Identify feature types
    numerical_features = ['age', 'pincode', 'num_dependents', 'estimated_salary', 'calls_made', 'sms_sent', 'data_used']
    categorical_features = ['telecom_partner', 'gender', 'state', 'city']
    
    print("Building pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ])
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training pipeline...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating pipeline...")
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Ensure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    pipeline_path = os.path.join(MODEL_DIR, 'random_forest_pipeline.pkl')
    joblib.dump(pipeline, pipeline_path)
    print(f"Pipeline saved to {pipeline_path}")

if __name__ == '__main__':
    main()
