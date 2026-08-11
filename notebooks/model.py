#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[4]:


# 1. Load processed data
X_train = pd.read_csv('../data/processed/X_train.csv')
y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
X_test = pd.read_csv('../data/processed/X_test.csv')
y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()


# In[5]:


# 2. Identify feature types
numerical_features = [col for col in ['age', 'estimated_salary', 'calls_made', 'sms_sent', 'data_used'] if col in X_train.columns]
categorical_features = [col for col in X_train.columns if col not in numerical_features]

# 3. Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# In[ ]:


# 4. Build full pipeline
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42))
])


# In[5]:


# 5. Train
clf.fit(X_train, y_train)


# In[ ]:


# 7. Save model and preprocessor
joblib.dump(clf.named_steps['classifier'], '../models/random_forest_classifier.pkl')
joblib.dump(clf.named_steps['preprocessor'], '../models/preprocessor.joblib')


# In[8]:


# 6. Evaluate (optional)
from sklearn.metrics import classification_report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

