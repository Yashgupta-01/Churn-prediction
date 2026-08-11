# 📞 TCS Interview Q&A — Telecom Churn Prediction Project
### Easy Level | 30 Questions | STAR Method | WHY · HOW · WHEN · WHAT

---

## 🔷 SECTION 1: Project Overview

---

### Q1. Tell me about your project. What is Telecom Churn Prediction?

**Answer (STAR):**

**Situation:** In the telecom industry, losing customers (churn) is a major revenue problem. Retaining an existing customer is 5× cheaper than acquiring a new one.

**Task:** I built a machine learning system to predict which telecom customers are likely to churn, so the business can take proactive retention actions.

**Action:**
- Collected a dataset of ~2.4 lakh telecom customer records with features like `age`, `gender`, `telecom_partner` (Airtel, BSNL, Reliance Jio, Vodafone), `calls_made`, `sms_sent`, `data_used`, and `estimated_salary`.
- Performed EDA, data preprocessing, trained a Random Forest Classifier, and deployed it as a Streamlit dashboard.

**Result:** A working end-to-end churn prediction system where a business user can input customer data and get real-time churn probability.

---

### Q2. What is the dataset you used? Describe its features.

**Answer:**

- **Source:** `telecom_churn.csv` — ~2,43,554 customer records
- **Features:**

| Column | Type | Description |
|---|---|---|
| `customer_id` | int | Unique ID |
| `telecom_partner` | categorical | Airtel, BSNL, Jio, Vodafone |
| `gender` | categorical | M / F |
| `age` | int | Customer age (18–74) |
| `state` | categorical | Indian state |
| `city` | categorical | City name |
| `pincode` | int | Area code |
| `date_of_registration` | date | When customer joined |
| `num_dependents` | int | Number of dependents |
| `estimated_salary` | float | Annual salary estimate |
| `calls_made` | int | Monthly calls |
| `sms_sent` | int | Monthly SMS |
| `data_used` | float | Monthly data in MB/GB |
| `churn` | int | **Target** — 0 = stayed, 1 = churned |

- **Why these features?** Usage metrics (calls, SMS, data) directly indicate engagement. Low engagement often correlates with churn.

---

### Q3. Why did you choose this problem for your project?

**Answer:**

- **Business Relevance:** Churn prediction is one of the most demanded ML use cases in telecom, banking, and e-commerce.
- **Real-world Data:** The dataset closely mimics real Indian telecom data with 4 operators and pan-India coverage.
- **Interview Value:** It covers the entire ML pipeline — EDA → preprocessing → modelling → deployment — making it comprehensive to present.
- **Impact:** Correctly predicting churn helps save lakhs in customer acquisition costs.

---

## 🔷 SECTION 2: EDA (Exploratory Data Analysis)

---

### Q4. What did you do during EDA? What insights did you find?

**Answer (STAR):**

**Situation:** Raw data always has hidden patterns and quality issues.

**Task:** Understand the data distribution, spot anomalies, and find features most related to churn.

**Action:**
1. **Shape check** — 2,43,554 rows × 14 columns
2. **Null check** — Verified no missing values
3. **Data type audit** — `date_of_registration` needed parsing
4. **Outlier detection** — Found **negative values** in `calls_made`, `sms_sent`, and `data_used` (e.g., -361, -3) — clear data quality issues
5. **Class distribution** — Checked churn ratio (important for imbalance detection)
6. **Correlation analysis** — Numerical feature correlation with target
7. **Categorical analysis** — Churn rate per `telecom_partner`, `gender`, `state`

**Key Insights:**
- Some customers had **negative usage values** — likely data entry errors
- Dataset spans multiple years (starting 2020-01-01) — `date_of_registration` can be used to derive customer tenure
- Churn varied across telecom partners

---

### Q5. What errors or anomalies did you find in the data?

**Answer:**

| Anomaly | Column | Example | Action Taken |
|---|---|---|---|
| Negative values | `calls_made` | -1, -10 | Treated as data errors, clipped or removed |
| Negative values | `sms_sent` | -4, -2 | Same treatment |
| Negative values | `data_used` | -361, -492 | Same treatment |
| High cardinality | `state`, `city` | 28+ states, 8 cities | Encoded carefully |
| Unused column | `pincode`, `customer_id` | — | Dropped before modelling |

**Why it matters:** Negative usage values can confuse the model. A customer can't make -5 calls — this is a **data quality issue** that must be handled before training.

---

### Q6. How did you handle class imbalance (if any)?

**Answer:**

- **What is class imbalance?** When one class (churn=0) heavily outnumbers the other (churn=1), the model can simply predict "0" always and still get high accuracy — but be useless.
- **How to detect:** Using `df['churn'].value_counts()` and plotting the distribution.
- **Common solutions:**
  - **SMOTE** (Synthetic Minority Oversampling Technique) — creates synthetic minority samples
  - **Class weights** — `class_weight='balanced'` in sklearn
  - **Undersampling** — reduce majority class
  - **Evaluation metric shift** — use F1-Score, ROC-AUC instead of Accuracy
- **In this project:** We used appropriate evaluation metrics (accuracy + probability) and Random Forest which handles mild imbalance naturally.

---

## 🔷 SECTION 3: Data Preprocessing

---

### Q7. What preprocessing steps did you perform?

**Answer:**

1. **Drop irrelevant columns:** `customer_id`, `pincode` — no predictive value
2. **Date feature engineering:** From `date_of_registration`, extract `customer_tenure_days` to capture loyalty
3. **Encoding categoricals:**
   - `gender` → Label Encoding (binary: M=1, F=0)
   - `telecom_partner`, `state`, `city` → One-Hot Encoding or Target Encoding
4. **Handling negatives:** Clip `calls_made`, `sms_sent`, `data_used` to 0 minimum
5. **Scaling:** StandardScaler for numerical features (important for distance-based models, less critical for tree models)
6. **Train-Test Split:** 80-20 split using `train_test_split` with `random_state=42` for reproducibility

**Why a preprocessor pipeline?** The `preprocessor.pkl` file in the dashboard ensures the **same transformations** applied during training are applied at inference — preventing data leakage and inconsistency.

---

### Q8. What is a preprocessing pipeline and why did you save it?

**Answer:**

- **What:** A `sklearn.pipeline.Pipeline` chains preprocessing steps (encoding, scaling) with the model into one object.
- **Why save it:** At prediction time (in the Streamlit dashboard), user input must be transformed **exactly the same way** as training data. If we only save the model but not the preprocessor, we'd need to manually redo all transformations — which is error-prone.
- **How:** `joblib.dump(preprocessor, 'preprocessor.pkl')` saves it; `joblib.load('preprocessor.pkl')` restores it.
- **Error faced:** If the preprocessor and model are trained on different feature orders, `transform()` will throw a **feature mismatch error** at runtime.

---

### Q9. What is the difference between Label Encoding and One-Hot Encoding? When did you use which?

**Answer:**

| | Label Encoding | One-Hot Encoding |
|---|---|---|
| **What** | Assigns integer 0, 1, 2... | Creates binary column per category |
| **When to use** | Ordinal data or binary (gender) | Nominal data with no order |
| **Problem** | Implies false ordering (Airtel=0 < BSNL=1?) | Increases dimensionality |
| **Used for** | `gender` (M/F) | `telecom_partner`, `state`, `city` |

**Why it matters:** Using Label Encoding on `telecom_partner` would incorrectly suggest Airtel < BSNL < Jio < Vodafone, biasing tree splits.

---

## 🔷 SECTION 4: Model Building

---

### Q10. Why did you choose Random Forest as your model?

**Answer:**

- **What is Random Forest?** An ensemble of Decision Trees trained on random subsets of data and features, predictions aggregated by majority vote.
- **Why Random Forest for churn?**
  1. Handles mixed data types (numerical + categorical) well
  2. Robust to outliers and missing values
  3. Provides **feature importance** — tells which features drive churn
  4. Less prone to overfitting vs. a single Decision Tree
  5. Works well even without extensive hyperparameter tuning
- **Alternatives considered:** Logistic Regression (simpler baseline), XGBoost (more powerful but complex), SVM (slow on large data)

---

### Q11. What is overfitting and how did you prevent it?

**Answer:**

- **What:** Model learns training data too well — performs great on train set but poorly on unseen test data.
- **Signs:** Train accuracy = 99%, Test accuracy = 72% → overfitting
- **Prevention in Random Forest:**
  - `max_depth` — limits tree depth
  - `min_samples_split` — minimum samples to split a node
  - `n_estimators` — more trees reduce variance
  - `max_features` — random feature selection per split
- **General prevention:** Cross-validation, train-test split, regularization

---

### Q12. How did you split the data and why 80-20?

**Answer:**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **80-20 split:** 80% training, 20% testing — a standard industry practice
- **`stratify=y`:** Ensures churn proportion is preserved in both splits — critical for imbalanced datasets
- **`random_state=42`:** Ensures reproducibility — same split every run
- **Why not 70-30?** With ~2.4L records, 80% gives ample training data while 20% (~48K) is sufficient for reliable evaluation

---

### Q13. What evaluation metrics did you use and why?

**Answer:**

| Metric | Formula | Why Used |
|---|---|---|
| **Accuracy** | (TP+TN)/(Total) | Overall correctness |
| **Precision** | TP/(TP+FP) | Cost of false alarms |
| **Recall** | TP/(TP+FN) | Cost of missing churners |
| **F1-Score** | 2×P×R/(P+R) | Balance of precision & recall |
| **ROC-AUC** | Area under ROC curve | Model's discriminating ability |

**Why not just accuracy?** If 90% customers don't churn, predicting "no churn" always gives 90% accuracy — but the model is useless. **Recall for churn class** is most important — we'd rather flag a non-churner than miss a churner.

---

### Q14. What is the confusion matrix and what does it tell you?

**Answer:**

```
              Predicted: No    Predicted: Yes
Actual: No  |   TN (good)   |   FP (false alarm)  |
Actual: Yes |   FN (miss!)  |   TP (correct!)     |
```

- **TN (True Negative):** Correctly predicted customer stays → great
- **TP (True Positive):** Correctly predicted churn → great, action can be taken
- **FP (False Positive):** Predicted churn but customer stays → wastes retention budget
- **FN (False Negative):** Predicted no churn but customer churns → worst case — revenue lost

**Business priority:** Minimize **FN** (missed churners). So we optimize **Recall**.

---

## 🔷 SECTION 5: Feature Engineering & Importance

---

### Q15. What feature engineering did you perform?

**Answer:**

1. **Customer Tenure:** `date_of_registration` → `tenure_days = (today - registration_date).days` — longer tenure often means loyal customer
2. **Usage ratios:** Could engineer `calls_per_day`, `data_per_call` to capture engagement patterns
3. **Dropped:** `customer_id` (identifier, not a feature), `pincode` (too granular, high cardinality)
4. **Date parsing:** `pd.to_datetime('date_of_registration')` to enable date arithmetic

**Why feature engineering matters:** Raw features often don't capture relationships. Tenure is more meaningful than a raw date string.

---

### Q16. How do you find which features are most important for churn prediction?

**Answer:**

```python
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print(feat_imp)
```

- Random Forest provides built-in **feature importance** scores
- **Likely top features:** `estimated_salary`, `data_used`, `calls_made`, `age`, `num_dependents`
- **Why important?** Helps business focus on the right signals and helps reduce model complexity by dropping low-importance features

---

## 🔷 SECTION 6: Dashboard (Streamlit)

---

### Q17. How does your Streamlit dashboard work?

**Answer:**

**Flow:**
1. **Load artifacts** → `joblib.load()` loads `random_forest_classifier.pkl` and `preprocessor.joblib`
2. **User Input** → Streamlit widgets (`st.number_input`, `st.selectbox`) capture customer data
3. **DataFrame creation** → Input dict → `pd.DataFrame([input_features])`
4. **Preprocessing** → `preprocessor.transform(input_df)` applies same transforms as training
5. **Prediction** → `model.predict()` for label, `model.predict_proba()` for probability
6. **Display** → `st.success()` / `st.info()` show churn verdict and probability %

```python
churn_pred = model.predict(X_processed)[0]       # 0 or 1
churn_prob = model.predict_proba(X_processed)[0][1]  # probability of churn
```

---

### Q18. What errors did you face in the dashboard and how did you fix them?

**Answer:**

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError` for model | Wrong relative path (`../models/`) when running from dashboard/ dir | Use `os.path.abspath()` or place model in same dir |
| `Feature mismatch` | Input DataFrame columns don't match training columns | Ensure input dict has exact same keys as training features |
| `ValueError: transform expects n features` | Preprocessor saved without fitting on all features | Re-train and re-save preprocessor after finalizing features |
| Model returns wrong probabilities | `predict_proba()[0][1]` vs `[0][0]` confused | Index `[1]` = churn probability, `[0]` = no-churn |
| Streamlit re-runs on every widget change | Default Streamlit behavior | Wrap prediction in `if st.button()` block |

---

### Q19. Why did you use `@st.cache_resource` for loading the model?

**Answer:**

- **What:** `@st.cache_resource` is a Streamlit decorator that caches the result of a function across reruns.
- **Why:** Streamlit re-executes the entire script on every user interaction. Without caching, the model would be **reloaded from disk on every button click** — extremely slow for large models.
- **When:** Applied to `load_artifacts()` which loads both model and preprocessor.
- **Result:** Model is loaded once into memory and reused — faster response time.

---

## 🔷 SECTION 7: ML Concepts (Deep Dive)

---

### Q20. Explain how a Decision Tree works.

**Answer:**

- **What:** A tree structure where each node splits data based on a feature threshold to maximize **information gain** (or minimize **Gini impurity**).
- **How:**
  1. Start with all data at root
  2. Find the feature & threshold that best separates classes
  3. Split data → repeat recursively
  4. Stop when leaf nodes are pure or depth limit reached
- **Gini Impurity:** `G = 1 - Σ(p²)` — measures how mixed a node is. Goal = minimize Gini.
- **Limitation:** Prone to overfitting → solved by **Random Forest** (ensemble of trees)

---

### Q21. What is the difference between `predict()` and `predict_proba()`?

**Answer:**

```python
model.predict(X)        # Returns [0, 1, 0, 1] — class labels
model.predict_proba(X)  # Returns [[0.8,0.2], [0.3,0.7]] — probabilities
```

- **`predict()`:** Hard classification — applies a threshold (default 0.5) to assign class
- **`predict_proba()`:** Soft output — gives probability for each class
- **In our dashboard:**
  - `predict()` → "Churn: Yes / No"
  - `predict_proba()[0][1]` → "Churn Probability: 73%"
- **Why probability matters:** Business can prioritize high-probability churners for outreach rather than treating all predicted churners equally.

---

### Q22. What is cross-validation and did you use it?

**Answer:**

- **What:** Instead of a single train-test split, data is split into `k` folds. Model trains on k-1 folds and validates on 1 fold, repeated k times.
- **Why:** More robust evaluation — reduces luck of a single split
- **Common type:** K-Fold (k=5 or k=10), Stratified K-Fold (preserves class ratio)

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(scores.mean())  # average F1 across 5 folds
```

- **In this project:** The `X_train.csv` and `X_test.csv` already saved suggest a single split was used, but cross-validation is recommended to validate model stability.

---

### Q23. What is ROC-AUC? How do you interpret it?

**Answer:**

- **ROC Curve:** Plots **True Positive Rate (Recall)** vs **False Positive Rate** at various thresholds
- **AUC (Area Under Curve):**
  - AUC = 1.0 → Perfect model
  - AUC = 0.5 → Random guessing (useless)
  - AUC = 0.8 → Good model (80% chance of ranking a churner higher than a non-churner)
- **Why use it for churn?** Threshold-independent — evaluates model across all decision boundaries, better than single accuracy metric for imbalanced datasets.

```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
```

---

## 🔷 SECTION 8: Errors Faced During the Project

---

### Q24. What common errors did you face during model training?

**Answer:**

| Error | When | Cause | Fix |
|---|---|---|---|
| `ValueError: could not convert string to float` | During `fit()` | Forgot to encode categorical columns | Apply `LabelEncoder` / `OneHotEncoder` first |
| `MemoryError` | Large dataset processing | Loading full dataset without chunking | Use `pd.read_csv(chunksize=)` or drop unused columns |
| `KeyError: column not found` | During `transform()` | Column name mismatch between train & test | Standardize column names before split |
| `ConvergenceWarning` | Logistic Regression | Too few iterations | Increase `max_iter=1000` |
| `Pickle protocol error` | Loading old `.pkl` on newer Python | Version mismatch | Retrain and resave in same environment |

---

### Q25. What error occurred with negative values in the dataset and how did you handle it?

**Answer (STAR):**

**Situation:** During EDA, I noticed some rows had `calls_made = -1`, `data_used = -361`, `sms_sent = -4` — physically impossible values.

**Task:** Clean these anomalies without losing too many data points.

**Action:**
```python
# Option 1: Clip negatives to 0
df['calls_made'] = df['calls_made'].clip(lower=0)
df['sms_sent'] = df['sms_sent'].clip(lower=0)
df['data_used'] = df['data_used'].clip(lower=0)

# Option 2: Drop rows where any usage is < 0
df = df[(df['calls_made'] >= 0) & (df['sms_sent'] >= 0) & (df['data_used'] >= 0)]
```

**Result:** Clean numerical features that don't confuse the model with impossible values.

**Why clip over drop?** With 2.4L rows, dropping rows with slight negatives is acceptable, but clipping retains more data — preferred when the negative value is likely a minor recording error.

---

## 🔷 SECTION 9: Behavioral / HR Questions

---

### Q26. Describe a challenge you faced in this project and how you overcame it. (STAR)

**Answer:**

**Situation:** After building the Streamlit dashboard, the model would throw a `FileNotFoundError` when loading the `.pkl` file, even though the file existed.

**Task:** Fix the path resolution issue so the app runs correctly regardless of where it's launched from.

**Action:**
- The issue was that `os.path.join('..', 'models', 'random_forest_classifier.pkl')` resolves relative to the **current working directory**, not the script location.
- Fixed by using: `os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')` — always resolves relative to the script file.
- Alternatively, placed the model file inside the `dashboard/` folder to simplify paths.

**Result:** Dashboard ran successfully with no path errors from any directory.

---

### Q27. How did you ensure your model doesn't have data leakage?

**Answer:**

- **What is data leakage?** When information from the test set "leaks" into training — model appears great on test but fails in production.
- **Common sources:**
  1. Fitting preprocessor (scaler, encoder) on full dataset before split
  2. Using target-correlated features that wouldn't exist at prediction time
- **How I prevented it:**
  1. Fit preprocessor **only on X_train**, then transform X_test separately
  2. Did not use `customer_id` or any post-churn information
  3. Used `Pipeline` to chain preprocessing + model — prevents leakage automatically

```python
# WRONG — leakage!
scaler.fit(X)  # sees test data
X_train_scaled = scaler.transform(X_train)

# CORRECT — no leakage
scaler.fit(X_train)  # only training data
X_test_scaled = scaler.transform(X_test)
```

---

### Q28. Why did you use joblib instead of pickle to save the model?

**Answer:**

- **pickle:** Python's built-in serialization — works but slow for large numpy arrays
- **joblib:** Optimized for numpy arrays — up to **10× faster** for scikit-learn models
- **How:**
```python
import joblib
joblib.dump(model, 'model.pkl')         # save
model = joblib.load('model.pkl')        # load
```
- **When to use joblib:** Always preferred for sklearn models, pipelines, and any object containing large arrays
- **Error faced:** Using `pickle` on large Random Forest with 100 trees was slow — switched to `joblib` for faster save/load.

---

### Q29. How would you improve this project further?

**Answer:**

1. **Better model:** Try **XGBoost** or **LightGBM** — generally outperform Random Forest on tabular data
2. **Hyperparameter tuning:** `GridSearchCV` or `RandomizedSearchCV` to find optimal `n_estimators`, `max_depth`
3. **Feature engineering:** Customer tenure from `date_of_registration`, usage trend over time
4. **Handle imbalance explicitly:** SMOTE or class weights
5. **Explainability:** Add **SHAP values** to dashboard — show WHY a customer is predicted to churn
6. **Monitoring:** Track model drift over time — retrain when performance degrades
7. **API:** Expose model via **FastAPI** for production use instead of just a dashboard
8. **Better UI:** Add charts, customer segment analysis, historical churn trends

---

### Q30. What is the business impact of your churn prediction model?

**Answer (STAR):**

**Situation:** Telecom companies lose 15–25% of their subscriber base annually to churn, each customer worth thousands of rupees in annual revenue.

**Task:** Quantify why the ML model adds business value.

**Action & Impact:**
- **Early Warning System:** Model flags at-risk customers 30–60 days before churn → retention team can intervene with offers
- **Cost Reduction:** Retaining a customer costs ~20% of acquiring a new one
- **Targeted Marketing:** Instead of mass discounts, only high-risk customers get retention offers → saves budget
- **Probability Score:** `predict_proba()` allows tiered response — very high risk (>80%) gets premium retention offers; moderate risk (50-80%) gets standard offers

**Result:** A 10% improvement in churn prediction recall could retain hundreds of additional customers per month, translating to significant revenue protection.

---

## 📋 Quick Reference Cheat Sheet

| Topic | Key Point |
|---|---|
| Dataset | 2,43,554 rows, 14 features, binary target (`churn`) |
| Operators | Airtel, BSNL, Reliance Jio, Vodafone |
| Key anomaly | Negative values in `calls_made`, `sms_sent`, `data_used` |
| Model | Random Forest Classifier |
| Saved artifacts | `random_forest_classifier.pkl`, `preprocessor.joblib` |
| Dashboard | Streamlit with `@st.cache_resource` |
| Key metric | ROC-AUC + F1-Score (not just accuracy) |
| Leakage prevention | Fit preprocessor only on X_train |
| Serialization | `joblib.dump()` / `joblib.load()` |
| Main error | File path resolution in dashboard |

---

> **Tip for TCS Interview:** Always lead with the business problem before jumping into technical details. TCS values clear communication and practical thinking. Use numbers (2.4 lakh rows, 4 telecom partners, 80-20 split) to sound confident and specific.
