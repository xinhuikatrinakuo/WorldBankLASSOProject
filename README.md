<h1>Household Poverty Status Prediction Using LASSO</h1>

<h2>Description</h2>

Identifying households in poverty is essential for providing effective assistance, but traditional surveys are expensive and time-consuming. The World Bank aims to identify key factors influencing household poverty to improve data collection efficiency. The goal of this project is to predict household poverty status (`poor`: True/False) using machine learning and to select the most relevant features with **LASSO (L1-regularized) Logistic Regression**. LASSO shrinks weak predictors to zero, highlights the most important features, prevents overfitting, and allows the World Bank to reduce the number of survey questions while maintaining accuracy.
<br />


<h2>Data</h2>

Analysis uses two datasets: a **training set (6,578 households)** and a **test set (1,625 households)**, each containing 346 features. Each observation corresponds to a unique household and is labeled with a binary poverty indicator (`Poor`). 

**Features include:**
- Categorical indicators (e.g., ownership of items such as bar soap, cooking oil, matches, salt)
- Numeric measures (e.g., number of working cell phones, number of rooms in the household)

<p align="center">
<img width="457" height="258" alt="Screenshot 2025-09-12 at 3 46 39 PM" src="https://github.com/user-attachments/assets/1817f46f-9414-45bb-b26e-f1ba15e5a2f4" />
<br>
<p align="left">
All variables are encoded as random character strings, so our focus is on predictive power rather than interpretability. The target distribution is balanced: 

- Non-poor: 54.6%
- Poor: 45.4%

This balance means no special class weighting is required.
```python
train_data.info()
# RangeIndex: 6578 entries, 346 columns
# dtypes: bool(1), int64(5), object(340)

test_data.info()
# RangeIndex: 1625 entries, 346 columns
# dtypes: bool(1), int64(5), object(340)
```
<h3>Data Preprocessing</h3>

1. Encode categorical variables
```python
from sklearn.preprocessing import LabelEncoder

# Identify categorical columns (object type)
categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()

# Apply label encoding to categorical columns in both train and test datasets
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = test_data[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

# Confirm encoding success by displaying first few rows
train_data.head()
```
<img width="545" height="162" alt="Screenshot 2025-09-24 at 5 09 32 PM" src="https://github.com/user-attachments/assets/4b57850e-2146-4bd5-b25a-791f9aab20f8" />

This ensures all categorical variables are numeric for modeling.

2. Align features between train and test sets:
```python
# Ensure train and test datasets have the same features

# Find common columns in both datasets
common_features = list(set(train_data.columns) & set(test_data.columns))

# Keep only the common features in both datasets
train_data = train_data[common_features]
test_data = test_data[common_features]

# Confirm feature alignment
train_data.shape, test_data.shape
```
3. Standardize features:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Define features (X) and target variable (y)
X_train = train_data.drop(columns=['id', 'poor'], errors='ignore')
y_train = train_data['poor']

# Standardize features for Lasso
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
```

<h2>Model: LASSO Logistic Regression</h2>

LASSO was selected because it:
- Selects the most relevant features by shrinking weaker predictors to zero.
- Prevents overfitting via the L1 penalty.
- Simplifies interpretation by focusing on strong predictors.

**Training and feature selection:**
```python
# Apply Lasso (L1 Regularization) Logistic Regression
lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
lasso.fit(X_scaled_train, y_train)

# Get the selected features (non-zero coefficients)
selected_features = X_train.columns[np.abs(lasso.coef_)[0] > 0]
```
**Fixing test set mismatches and standardizing:**
```python
from sklearn.metrics import log_loss

# Fix feature mismatch error
## Step 1: Ensure both datasets have the exact same feature set

# Find common features in both train and test sets
common_features = set(X_train.columns) & set(X_test.columns)

# Find missing features in the test set
missing_features = set(X_train.columns) - set(X_test.columns)

# Add missing columns to the test set with default values (0)
for col in missing_features:
    X_test[col] = 0

# Reorder columns in test set to match train set
X_test = X_test[X_train.columns]

## Step 2: Re-standardize both datasets
X_scaled_test = scaler.transform(X_test)
```
**Predict and evaluate Log Loss:**
```python
## Step 3: Retrain Lasso on aligned datasets
lasso.fit(X_scaled_train, y_train)

## Step 4: Predict probabilities on the test set
y_pred_proba = lasso.predict_proba(X_scaled_test)[:, 1]  # Probability of being poor

## Step 5: Compute Log Loss
log_loss_score = log_loss(test_data['poor'], y_pred_proba)

# Display final Log Loss score
print("Log Loss Score:", log_loss_score)
```
Log Loss Score: 0.2867312867007622

<h2>Results and Conclusion</h2>

**Performance:**
<br>
Log Loss = 0.2867, indicating strong predictive accuracy. It also means that the LASSO model is highly accurate in predicting the probability of a household being in poverty. 

The World Bank can 
- trust these predictions to guide data collection and aid distribution. So instead of surveying each household, the World Bank can focus on high-probability household first to help.
- use LASSO to identify which predictors to ask if they need to go to the households for survey questions to reduce survey costs.

```python
import matplotlib.pyplot as plt
import numpy as np

# Extract Lasso coefficients and corresponding feature names
lasso_coefficients = lasso.coef_[0]
feature_names = X_train.columns

# Sort features by absolute coefficient values
sorted_indices = np.argsort(np.abs(lasso_coefficients))[::-1]
sorted_coefficients = lasso_coefficients[sorted_indices]
sorted_features = np.array(feature_names)[sorted_indices]

# Plot the top 20 most important features
plt.figure(figsize=(10, 6))
plt.barh(sorted_features[:20], sorted_coefficients[:20], color="blue")
plt.xlabel("Lasso Coefficients")
plt.ylabel("Features")
plt.title("Top 20 Important Features Selected by Lasso")
plt.gca().invert_yaxis()  # Invert y-axis for readability
plt.show()
```
<p align="center">
<img width="895" height="532" alt="Screenshot 2025-09-24 at 5 28 19 PM" src="https://github.com/user-attachments/assets/34acc0bd-263a-41aa-b2d6-56caf10fd49f" />
<p align="left">

This bar chart visualizes the Top 20 Most Important Features selected by LASSO Regularization (L1 penalty).

1. **Feature Importance**:
- Each bar represents a feature from the dataset.
- The length of the bar (LASSO coefficient) indicates how strongly the feature impacts poverty prediction.

2. **Positive vs. Negative Coefficients**
- Positive coefficients (Right side) → Increase the likelihood of poverty.
- Negative coefficients (Left side) → Reduce the likelihood of poverty.
- If a feature has a large absolute coefficient, it has a strong influence on predictions.

→ We can see that Feature `TiwRslOh` has the largest negative coefficient meaning that it strongly reduces the likelihood of poverty; also, `GIMIxlmv` has a large negative coefficient and reduces the likelihood of poverty. The second and third large positive coefficients are `gwhBRami` and `HmDAlkAH` meaning that they strongly increase the likelihood of poverty. etc.

```python
# Train a Lasso (L1-regularized) Logistic Regression model
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
lasso_model.fit(X_scaled_train, y_train)

# Get feature importance (absolute values of coefficients)
lasso_feature_importance = abs(lasso_model.coef_[0])

# Create a DataFrame to store Lasso feature importance scores
lasso_feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': lasso_feature_importance})
lasso_feature_importance_df = lasso_feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top 20 most important features based on Lasso
lasso_feature_importance_df.head(20)
```
This LASSO feature importance table matches the bar chart above and shows the absolute value of the LASSO coefficient. 
- Higher values → Stronger impact on poverty prediction.
- Lower values → Less influence, but still significant.

→ The World Bank should focus on collecting data for these key features to effectively predict poverty.

**Tradeoff:** Reducing the number of features simplifies the survey and speeds up modeling, but it may slightly reduce the richness of information captured. However, LASSO ensures that the most important predictors are retained, so the loss in predictive power is minimal compared to the gains in efficiency and interpretability.

→ This means the World Bank can use fewer survey questions without losing prediction accuracy. The sacrifice is worth it.
