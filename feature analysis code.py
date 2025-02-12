# load libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# load data
df = pd.read_csv('/Users/itsalthomas/Desktop/churn_modeling.csv')
df


# ensuring no missing values - data is clean and ready to go
df.sum().isnull()


# drop unnecessary columns 
X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = df['Exited'] # target variable (1= churn, 0 = no churn)


# encode categorical values to binaries
X = pd.get_dummies(X, drop_first = True)
# train test split using 70/30% in training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123, stratify=y)


# standarize numerical features
scaler = StandardScaler()       
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# train logistic regression model with balance weighted classes
log_model = LogisticRegression(class_weight = 'balanced', random_state = 123)
log_model.fit(X_train, y_train)


# extract feature importance (logistic regression coefficients)
feature_importance = pd.DataFrame({
    'Feature': X.columns, 
    'Coefficient': log_model.coef_[0]
})


# sort by absolute coefficient values
feature_importance['Absolute_Value'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by = 'Absolute_Value', ascending = False).drop(columns = ['Absolute_Value'])

# display feature importance
print(feature_importance)


# visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=['red' if x > 0 else 'blue' for x in feature_importance['Coefficient']])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance (Logistic Regression)')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()

# the higher the absolute coefficient is, the more important that feature will be to deterimine whether or not a customer will churn