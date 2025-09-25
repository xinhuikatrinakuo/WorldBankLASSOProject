<h1>Predicting Household Poverty Status Using Rapid Survey Data</h1>

<h2>Description</h2>

Identifying households in poverty is essential for providing effective assistance, but traditional surveys are expensive and time-consuming. The World Bank aims to identify key factors influencing household poverty to improve data collection efficiency. The goal is this project is to predict household poverty status (`poor`: True/False) using machine learning and to select the most relevant features with **LASSO (L1-regularized) Logistic Regression**. LASSO shrinks weak predictors to zero, highlights the most important features, prevents overfitting, and allows the World Bank to reduce the number of survey questions while maintaining accuracy.
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
# Display the top selected features
dataframe=pd.DataFrame(selected_features, columns=["Feature"])
print(dataframe)

# Prepare the test dataset using selected features
X_test = test_data[selected_features]
```
**Fixing test set mismatches and standardizing:**
```python
# Display the top selected features
dataframe=pd.DataFrame(selected_features, columns=["Feature"])
print(dataframe)

# Prepare the test dataset using selected features
X_test = test_data[selected_features]
```
<h2>Program walk-through:</h2>

<p align="center">
Launch the utility: <br/>
<img src="https://i.imgur.com/62TgaWL.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Select the disk:  <br/>
<img src="https://i.imgur.com/tcTyMUE.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Enter the number of passes: <br/>
<img src="https://i.imgur.com/nCIbXbg.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Confirm your selection:  <br/>
<img src="https://i.imgur.com/cdFHBiU.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Wait for process to complete (may take some time):  <br/>
<img src="https://i.imgur.com/JL945Ga.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Sanitization complete:  <br/>
<img src="https://i.imgur.com/K71yaM2.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Observe the wiped disk:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
</p>

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
