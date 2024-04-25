- 👋 Hi, I’m @jaydeb2023
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

auto = pd.read_excel('loan Presiction.xlsx')

auto.head()

auto.info()

auto.isnull().sum()

auto['Loan_Amount_log']=np.log (auto['LoanAmount'])
auto['Loan_Amount_log'].hist(bins=20)

auto.isnull().sum()

auto['Gender'].fillna(auto['Gender'].mode()[0],inplace=True)
auto['Married'].fillna(auto['Married'].mode()[0],inplace=True)
auto['Self_Employed'].fillna(auto['Self_Employed'].mode()[0],inplace=True)
auto['Dependents'].fillna(auto['Dependents'].mode()[0],inplace=True)

auto.LoanAmount=auto.LoanAmount .fillna(auto.LoanAmount.mean())
auto.Loan_Amount_log =auto.Loan_Amount_log .fillna(auto.Loan_Amount_log .mean())

auto['Loan_Amount_Term'].fillna(auto['Loan_Amount_Term'].mode()[0],inplace=True)
auto['Credit_History'].fillna(auto['Credit_History'].mode()[0],inplace=True)
auto.isnull().sum()

x=auto.iloc[:,np.r_[1:5,9:11,13:14]].values
y=auto.iloc[:,12].values

x

y

print("per of missing gender is %2f%%" %((auto['Gender'].isnull().sum()/auto.shape[0])*100))

print("number of people who take loan as group by gender:")
print(auto['Gender'].value_counts())
sns.countplot(x='Gender',data=auto,palette='Set1')

print("number of people who take loan as group by Dependents:")
print(auto['Dependents'].value_counts())
sns.countplot(x='Dependents',data=auto,palette='Set1')

print("number of people who take loan as group by Self_Employed:")
print(auto['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed',data=auto,palette='Set1')

print("number of people who take loan as group by LoanAmount:")
print(auto['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount',data=auto,palette='Set1')

print("number of people who take loan as group by Credit_History:")
print(auto['Credit_History'].value_counts())
sns.countplot(x='Credit_History',data=auto,palette='Set1')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import LabelEncoder
LabelEncoder_x=LabelEncoder()


import numpy as np

# Assuming x_train is a numpy array and LabelEncoder_x is already defined
for i in range(0, 5):
    # Convert the entire column to string type before encoding
    x_train[:, i] = x_train[:, i].astype(str)
    x_train[:, i] = LabelEncoder_x.fit_transform(x_train[:, i])

# Convert the last column to string type before encoding
x_train[:, -1] = x_train[:, -1].astype(str)
x_train[:, -1] = LabelEncoder_x.fit_transform(x_train[:, -1])



LabelEncoder_y=LabelEncoder()
y_train=LabelEncoder_y.fit_transform(y_train)

y_train

import numpy as np

# Assuming x_test is a numpy array and LabelEncoder_x is already defined
for i in range(0, 5):
    # Convert the entire column to string type before encoding
    x_test[:, i] = x_test[:, i].astype(str)
    x_test[:, i] = LabelEncoder_x.fit_transform(x_test[:, i])

# Convert the last column to string type before encoding
x_test[:, -1] = x_test[:, -1].astype(str)
x_test[:, -1] = LabelEncoder_x.fit_transform(x_test[:, -1])



LabelEncoder_y=LabelEncoder()
y_test=LabelEncoder_y.fit_transform(y_test)

y_test

from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming x_train and x_test are defined numpy arrays
ss = StandardScaler()

# Fit and transform the training data
x_train = ss.fit_transform(x_train)

# Ensure x_test is of a numeric type
x_test = x_test.astype(float)

# Check for NaNs or infinite values in the test data and handle them
if np.any(np.isnan(x_test)) or not np.all(np.isfinite(x_test)):
    x_test = np.nan_to_num(x_test)

# Transform the test data
x_test = ss.transform(x_test)

# Print the transformed x_train and x_test arrays
print("Transformed x_train:", x_train)
print("Transformed x_test:", x_test)



from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(x_train,y_train)

# Correct the import statement and the module name
from sklearn import metrics

# Assuming rf_clf is a trained random forest classifier, x_test and y_test are defined
y_pred = rf_clf.predict(x_test)

# Print the accuracy of the random forest classifier
print("Accuracy of random forest classifier is", metrics.accuracy_score(y_pred, y_test))

# Print the predicted values
print("Predicted values:", y_pred)


from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(x_train,y_train)

y_pred = nb_clf.predict(x_test)
print("acc of GaussianNB is % ", metrics.accuracy_score(y_pred,y_test))


y_pred

from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
label_encoder = LabelEncoder()

# Assuming 'y_train' contains the categorical variable with 'Yes' and 'No'
# Encode the categorical variable
y_train_encoded = label_encoder.fit_transform(y_train)

# Now 'y_train_encoded' contains 0s and 1s instead of 'Yes' and 'No'
# Fit the DecisionTreeClassifier with the encoded labels
dt_clf.fit(x_train, y_train_encoded)


from sklearn import metrics

# Assuming dt_clf is a trained Decision Tree Classifier, and x_test and y_test are defined
y_pred = dt_clf.predict(x_test)

# Print the accuracy of the Decision Tree Classifier
print("Accuracy of DT is", metrics.accuracy_score(y_pred, y_test))


y_pred

from sklearn.neighbors import KNeighborsClassifier

# Create a KNeighborsClassifier object
kn_clf = KNeighborsClassifier()

# Assuming x_train and y_train are defined
# Fit the classifier to the training data
kn_clf.fit(x_train, y_train)


# Assuming kn_clf is a trained KNeighborsClassifier, and x_test and y_test are defined
y_pred = kn_clf.predict(x_test)

# Print the accuracy of the KNeighborsClassifier
print("Accuracy of KN is", metrics.accuracy_score(y_pred, y_test))



