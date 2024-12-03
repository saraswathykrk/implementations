# %% [markdown]
# ## 1. Problem Statement
# Predict if a loan can be approved or not

# %% [markdown]
# ## 2. Hypothesis
# Analyse which all factors could impact the loan approval. 
# 
# Some of the factors that could impact the loan approval are:
# 1. Salary
# 2. Less loan amount - easily approved
# 3. Lesser monthly repayment amount - easily approved
# 4. Prior loans default rate
# 5. Loan term

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn

import matplotlib.pyplot as plt
import warnings

# %matplotlib inline
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 3. Download the dataset and get the train and test datasets

# %%
train = pd.read_csv("/Users/sara/Documents/GitHub/implementations/Loan Prediction/train_ctrUa4K.csv")
test = pd.read_csv("/Users/sara/Documents/GitHub/implementations/Loan Prediction/test_lAUu6dG.csv")

train_original = train.copy()
test_original = test.copy()

# %% [markdown]
# ## 4. Check the various columns in the train and test datasets and determine their datatypes and shape

# %%
train.columns

# %%
test.columns

# %%
train.head()

# %%
train.dtypes

# %% [markdown]
# We see there are three different formats of datatypes.
# 1. object: Means categorical
# 2. int64: Integer variables
# 3. float64: Numerical with decimal values

# %%
print("train shape:",train.shape)
print("test shape:",test.shape)

# %% [markdown]
# Train dataset has 614 rows and 13 columns, while test dataset has 367 rows and 12 columns (without the target variable - here loan status)

# %% [markdown]
# ## 5. Perform Univariate analysis of the dataset
# 
# Points to be noted:
# 
# There are 3 types of features involved:
# 1. Categorical features: Having a category - Gender, Married, Self_Employed, Credit_History, Loan_Status
# 2. Ordinal features: Categorical variables with some order involved - Dependents, Education, Property_Area
# 3. Numerical features: Having numerical values - ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
# 
# * For categorical variables, we can use frequency table (with normalization for percentages) and/or bar plots.
# * For numerical variables, probability density plots can be used to check the distribution - box plots or dist plots.

# %% [markdown]
# **Target Variable - Loan_Status**

# %%
print("Frequency Table of Loan_Status:",train.Loan_Status.value_counts())
print("-----------------------------------------------")
print("Percentage of Loan_Status:",train.Loan_Status.value_counts(normalize=True))
print("-----------------------------------------------")
train.Loan_Status.value_counts().plot.bar()


# %% [markdown]
# **Analysis: 422 (approx. 69%) loans out of the total 614 loans were approved.**

# %% [markdown]
# **Independent Categorical Variables**

# %%
plt.figure(1)

plt.subplot(221)
train.Gender.value_counts(normalize=True).plot.bar(figsize=(20,10),title="Gender")

plt.subplot(222)
train.Married.value_counts(normalize=True).plot.bar(title="Married")

plt.subplot(223)
train.Self_Employed.value_counts(normalize=True).plot.bar(title="Self_Employed")

plt.subplot(224)
train.Credit_History.value_counts(normalize=True).plot.bar(title="Credit_History")

plt.show()

# %% [markdown]
# **Inferences:**
# 
# **1. 80% of the applicants in the dataset are males.**
# 
# **2. Around 65% of the applicants are married.**
# 
# **3. About 15% of the applicants are self employed.**
# 
# **4. Around 85% of the applicants have repaid their debts.**

# %% [markdown]
# **Independent Ordinal Variables**

# %%
plt.figure(1)

plt.subplot(141)
train.Dependents.value_counts(normalize=True).plot.bar(figsize=(24,6),title="Dependents")

plt.subplot(142)
train.Education.value_counts(normalize=True).plot.bar(title="Education")

plt.subplot(143)
train.Property_Area.value_counts(normalize=True).plot.bar(title="Property_Area")

plt.subplot(144)
train.Loan_Amount_Term.value_counts(normalize=True).plot.bar(title="Loan_Amount_Term")

plt.show()

# %% [markdown]
# **Inferences:**
# 
# **1. Most of the applicants have no dependents.**
# 
# **2. About 80% of the applicants are graduates.**
# 
# **3. Most of the applicants are from semiurban area.**
# 
# **4. More than 80% of the loans are of 1 year terms.**

# %% [markdown]
# **Independent Numerical Variables**

# %% [markdown]
# **1. Applicant Income**

# %%
plt.figure(1)

plt.subplot(121)
sns.distplot(train.ApplicantIncome)

plt.subplot(122)
train.ApplicantIncome.plot.box(figsize=(16,5))

plt.show()

# %% [markdown]
# **Inferences:**
# 
# 1. Not normally distributed, right-skewed.
# 2. Box plot shows a lot of outliers - could be the income disparity in the society.
# 
# Let us check the Income of the applicant versus their education.

# %%
train.boxplot(column="ApplicantIncome", by="Education")
plt.suptitle("Applicant Income by Education")

# %% [markdown]
# **Inferences:**
# 
# From the above plot, it can be seen that more number of graduates have higher income which appear as outliers.

# %% [markdown]
# **2. Coapplicant Income**

# %%
plt.figure(1)

plt.subplot(121)
sns.distplot(train.CoapplicantIncome)

plt.subplot(122)
train.CoapplicantIncome.plot.box(figsize=(16,5))

plt.show()

# %% [markdown]
# **Inferences:**
# 
# 1. Not normally distributed, right-skewed - ranging from 0 to 5000.
# 2. Box plot shows a lot of outliers - could be the income disparity in the society - similar to ApplicantIncome.

# %% [markdown]
# **3. Loan Amount**

# %%
plt.figure(1)

plt.subplot(121)
sns.distplot(train.LoanAmount)

plt.subplot(122)
train.LoanAmount.plot.box(figsize=(16,5))

plt.show()

# %% [markdown]
# **Inferences:**
# 
# * Lot of outliers, but mostly normally distributed.

# %% [markdown]
# ## 6. Perform Bivariate analysis of the dataset with respect to the target variable
# 
# Recalling some of the points from our initial hypothesis:
# 
# 1. If loan amount is less, easy approval.
# 2. If prior loan repayment is good, easy approval.
# 3. If income is high, easy approval.
# 4. If loan term is less, easy approval.
# 5. If each term repayment amount is less, easy approval.
# 
# To verify if the above hypothesis is correct, lets do the bivariate analysis.

# %% [markdown]
# **Categorical / Ordinal Independent Variables vs Target Variable**

# %% [markdown]
# * Categorical Independent variables - Gender, Married, Dependents, Education, Self-Employed, Credit_History, Property_Area
# * Target Variable - Loan_Status

# %%
Gender = pd.crosstab(train.Gender, train.Loan_Status)
Married = pd.crosstab(train.Married, train.Loan_Status)
Dependents = pd.crosstab(train.Dependents, train.Loan_Status)
Education = pd.crosstab(train.Education, train.Loan_Status)
Self_Employed = pd.crosstab(train.Self_Employed, train.Loan_Status)
Credit_History = pd.crosstab(train.Credit_History, train.Loan_Status)
Property_Area = pd.crosstab(train.Property_Area, train.Loan_Status)

# %%
Gender.div(Gender.sum(1).astype(float),axis=0).plot.bar(stacked=True, figsize=(4,4))
Married.div(Married.sum(1).astype(float),axis=0).plot.bar(stacked=True, figsize=(4,4))
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

# %% [markdown]
# **Inferences:**
# 
# **1. Gender: Loan Status is more or less same for male and female applicants.**
# 
# **2. Married: More number of male applicants have loans approved.**
# 
# **3. Dependents: Status is similar for applications with 1 or 3+ dependents.**
# 
# **4. Education: More graduates have their loans approved.**
# 
# **5. Self-Employed: No particular inference for self-employment.**
# 
# **6. Credit_History: More number of people with credit_history 1 have their loans approved.**
# 
# **7. Property_Area: More approvals for Semiurbans areas.**

# %% [markdown]
# **Numerical Independent Variables vs Target Variable**

# %% [markdown]
# **1. ApplicantIncome**
# 
# We need to find the mean income of people with approved loans vs. mean income of people with unapproved loans and then compare.

# %%
train.ApplicantIncome.groupby(train.Loan_Status).mean().plot.bar()

# %% [markdown]
# **Inferences:**
# 
# * No differences can be seen in mean applicant income.
# 
# So, lets try making bins for the applicant income based on its values to analyze further.

# %%
# print(train.ApplicantIncome.max(),train.ApplicantIncome.min(), train.ApplicantIncome.mean())
# print(train.ApplicantIncome.value_counts())

bins = [0, 2500, 4000, 6000, 81000]
groups = ['low', 'average', 'high', 'very high']

train['Income_bin'] = pd.cut(train.ApplicantIncome, bins, labels = groups)

#Draw the plot

Income_bin = pd.crosstab(train.Income_bin, train.Loan_Status)
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot.bar(stacked=True, figsize = (4,4))

plt.xlabel("ApplicantIncome")
plt.ylabel("Percentage")

# %% [markdown]
# **Inferences:**
# 
# * From the above graph it is clear that the ApplicantIncome has no effect on the Loan approval, as we had initially thought in our hypothesis.

# %% [markdown]
# **2. Coapplicant Income**

# %%
train.CoapplicantIncome.groupby(train.Loan_Status).mean().plot.bar()

# %%
# print(train.CoapplicantIncome.max(),train.CoapplicantIncome.min(), train.CoapplicantIncome.mean())
# print(train.CoapplicantIncome.value_counts())

bins = [0, 1000, 3000, 4200]
groups = ['low', 'average', 'high']

train['Coapp_Income_bin'] = pd.cut(train.CoapplicantIncome, bins, labels = groups)

#Draw the plot

Coapp_Income_bin = pd.crosstab(train.Coapp_Income_bin, train.Loan_Status)
Coapp_Income_bin.div(Coapp_Income_bin.sum(1).astype(float),axis=0).plot.bar(stacked=True, figsize = (4,4))

plt.xlabel("CoapplicantIncome")
plt.ylabel("Percentage")

# %% [markdown]
# * The above plot shows that loan approval chances are high, when the coapplicant's income is less, which seems wierd. This could be because, many applicants do not have a coapplicant, so their income remains zero. To surpass this issue, we might have to take the Total Income to analyse this Income field.

# %% [markdown]
# **3. Total Income - new feature**

# %%
train['TotalIncome'] = train.ApplicantIncome + train.CoapplicantIncome
print(train.TotalIncome)


# %%
# print(train.TotalIncome.min(), train.TotalIncome.max())

bins = [0, 2500, 4000, 6000, 81000]
groups = ["Low", "Average", "High", "Very High"]

train["Total_Income_bin"] = pd.cut(train.TotalIncome, bins, labels = groups)
# print(train.Total_Income_bin)

#Draw the plot

Total_Income_bin = pd.crosstab(train.Total_Income_bin, train.Loan_Status)
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

plt.xlabel("Total_Income")
plt.ylabel("Percentage")

# %% [markdown]
# **Inferences:**
# 
# * We can see that less number of loans are getting approved, when the total income is less, compared to the average/high/very high income earning groups.

# %% [markdown]
# **4. Loan Amount**

# %%
train.LoanAmount.groupby(train.Loan_Status).mean().plot.bar()

# %%
# print(train.LoanAmount)
# print(train.LoanAmount.min(), train.LoanAmount.max())

bins = [0, 100, 200, 700]
groups = ["Low", "Average", "High"]

train["Loan_Amount_bin"] = pd.cut(train.LoanAmount, bins, labels = groups)
Loan_Amount_bin = pd.crosstab(train.Loan_Amount_bin, train.Loan_Status)
Loan_Amount_bin.div(Loan_Amount_bin.sum(1).astype(float),axis=0).plot(kind="bar", stacked = True, figsize =(4,4))

plt.xlabel("LoanAmount") 
plt.ylabel("Percentage")

# %% [markdown]
# **Inferences:**
# 
# * It can be seen from the above graph that the approved loans is higher for Low and Average Loan Amount as compared to high Loan Amount - which is what we had hypothesized initially.

# %% [markdown]
# **Find correlation between numerical variables and target variables**

# %% [markdown]
# * For this, we will drop all the bins we had created for the exploration part, as we do not need these bins in the dataset.
# 
# * We can change the dependents column from 3+ to 3 to make it a numerical variable.
# 
# * We can also convert the target variable into 0 and 1 to see if there is a correlation with numerical variables. We can replace N with 0 and Y with 1.

# %%
# train_hm = train.copy()
train_hm = train.drop(["Loan_ID", 
                        #   "Gender", "Married", "Education", "Self_Employed", "Property_Area",
                          "Income_bin", "Coapp_Income_bin", "Loan_Amount_bin", "Total_Income_bin", "TotalIncome"],axis=1)
train = train_hm.copy()

# %%
train_hm.head()

# %%
train_hm["Dependents"].replace("3+", "3", inplace=True)
# test_hm["Dependents"].replace("3+", "3", inplace=True)
train_hm["Loan_Status"].replace("N", 0, inplace=True)
train_hm["Loan_Status"].replace("Y", 1, inplace=True)
train_hm["Gender"].replace("Male", 0, inplace=True)
train_hm["Gender"].replace("Female", 1, inplace=True)
train_hm["Married"].replace("No", 0, inplace=True)
train_hm["Married"].replace("Yes", 1, inplace=True)
train_hm["Education"].replace("Graduate", 1, inplace=True)
train_hm["Education"].replace("Not Graduate", 0, inplace=True)
train_hm["Self_Employed"].replace("No", 0, inplace=True)
train_hm["Self_Employed"].replace("Yes", 1, inplace=True)
train_hm["Property_Area"].replace("Urban", 2, inplace=True)
train_hm["Property_Area"].replace("Rural", 0, inplace=True)
train_hm["Property_Area"].replace("Semiurban", 1, inplace=True)

# %%
print(train_hm.Dependents.unique())
print(train_hm.Loan_Status.unique())

# %% [markdown]
# **Now lets look at the correlation between the numerical variables.**

# %%
train_hm.columns

# %%
train_hm.dtypes

# %%
train_hm.head()

# %%
heatmap = train_hm.corr()

f, ax = plt.subplots(figsize = (9, 6))
sns.heatmap(heatmap, vmax = 0.8, square = True, cmap = "BuPu", annot=True)

# %% [markdown]
# **We can see that Applicant Income and Loan Amount are most correlated to each other as well as Credit History and Loan Status.**
# 
# **Loan Amount is also correlated to CoapplicantIncome.**

# %% [markdown]
# ## 7. Missing Values Imputation and Outlier treatment

# %%
train.isnull().sum()

# %% [markdown]
# ##### There are missing values in Gender, Married, Dependents, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History

# %% [markdown]
# We can fill the missing values in the following ways:
# 
# 1. For categorical variables - using mode
# 2. For numerical variables - using mean or median

# %% [markdown]
# ##### Categorical variables - Gender, Married, Dependents, Self_Employed, Credit_History

# %%
train.Gender.fillna(train.Gender.mode()[0], inplace=True)
train.Married.fillna(train.Married.mode()[0], inplace=True)
train.Dependents.fillna(train.Dependents.mode()[0], inplace=True)
train.Self_Employed.fillna(train.Self_Employed.mode()[0], inplace=True)
train.Credit_History.fillna(train.Credit_History.mode()[0], inplace=True)

# %% [markdown]
# ##### Loan Amount Term

# %%
train.Loan_Amount_Term.value_counts()

# %%
train.Loan_Amount_Term.fillna(train.Loan_Amount_Term.mode()[0], inplace=True)

# %% [markdown]
# ##### Loan Amount

# %%
train.LoanAmount.value_counts()

# %%
print(train.LoanAmount.min())
print(train.LoanAmount.mean())
print(train.LoanAmount.max())
print(train.LoanAmount.median())
print(train.LoanAmount.value_counts())

# %% [markdown]
# We will use median to fill missing values of Loan Amount as we had seen that it had many outliers.

# %%
train.LoanAmount.fillna(train.LoanAmount.median(), inplace=True)

# %%
train.isnull().sum()

# %% [markdown]
# ##### Similarly, lets fill test dataset missing values in the same manner

# %%
test.isnull().sum()

# %% [markdown]
# ##### Categorical

# %%
test.Gender.fillna(test.Gender.mode()[0], inplace=True)
test.Married.fillna(test.Married.mode()[0], inplace=True)
test.Dependents.fillna(test.Dependents.mode()[0], inplace=True)
test.Self_Employed.fillna(test.Self_Employed.mode()[0], inplace=True)
test.Credit_History.fillna(test.Credit_History.mode()[0], inplace=True)

# %% [markdown]
# ##### Loan Amount Term

# %%
test.Loan_Amount_Term.fillna(test.Loan_Amount_Term.mode()[0], inplace=True)

# %% [markdown]
# ##### Numerical - Loan Amount

# %%
test.LoanAmount.fillna(test.LoanAmount.median(), inplace=True)

# %%
test.isnull().sum()

# %% [markdown]
# ### Outlier treatment

# %% [markdown]
# Taking log transformation, can help remove skewness as it does not affect small values much but reduces larger values. So we get a distribution similar to a normal distribution.

# %%
train_hm.dtypes

# %% [markdown]
# 1. Applicant Income

# %%
sns.distplot(train_hm.ApplicantIncome)

# %%
train_hm.ApplicantIncome_log = np.log(train_hm.ApplicantIncome)
sns.distplot(train_hm.ApplicantIncome_log)

# %%
sns.distplot(test.ApplicantIncome)

# %%
test.ApplicantIncome_log = np.log1p(test.ApplicantIncome)
sns.distplot(test.ApplicantIncome_log)

# %% [markdown]
# 2. CoapplicantIncome

# %%
sns.distplot(train_hm.CoapplicantIncome)

# %%
train_hm.CoapplicantIncome_log = np.log1p(train_hm.CoapplicantIncome)
sns.distplot(train_hm.CoapplicantIncome_log)

# %%
sns.distplot(test.CoapplicantIncome)

# %%
test.CoapplicantIncome_log = np.log1p(test.CoapplicantIncome)
sns.distplot(test.CoapplicantIncome_log)

# %% [markdown]
# 3. Loan Amount

# %%
sns.distplot(train_hm.LoanAmount)

# %%
train_hm.LoanAmount_log = np.log(train_hm.LoanAmount)
sns.distplot(train_hm.LoanAmount_log)

# %%
sns.distplot(test.LoanAmount)

# %%
test.LoanAmount_log = np.log(test.LoanAmount)
# test.LoanAmount_log.min()
sns.distplot(test.LoanAmount_log)

# %%
test.LoanAmount_log.hist(bins=20)

# %% [markdown]
# ## 8. Model Building

# %%
train.head()

# %%
# train = train.drop('Loan_ID', axis=1)

# %%
test = test.drop('Loan_ID', axis=1)

# %%
X = train.drop('Loan_Status', axis=1)
y = train['Loan_Status']

# %%
X.head()

# %%
X = pd.get_dummies(X)
X.head()

# %%
train = pd.get_dummies(train)
test = pd.get_dummies(test)
train.head()

# %%
test.head()

# %%
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.3)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

# %%
y_pred_cv = model.predict(X_cv)

accuracy_score(y_cv, y_pred_cv)

# %%
y_pred_test = model.predict(test)
# y_pred_test

# %%
submission = pd.read_csv("sample_submission_49d68Cx.csv")
submission.head()

# %%
submission['Loan_Status'] = y_pred_test
submission['Loan_ID'] = test_original['Loan_ID']
submission.head()

# %%
submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)
submission.head()

# %%
pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('logistic.csv', index=False)

# %% [markdown]
# #### Using Stratified KFold

# %%
from sklearn.model_selection import StratifiedKFold

sum_score = 0
i = 1

kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))

    xtr, xvl = X.iloc[train_index], X.iloc[test_index]
    ytr, yvl = y.iloc[train_index], y.iloc[test_index]

    model = LogisticRegression(random_state=1)

    model.fit(xtr, ytr)

    y_pred_vl = model.predict(xvl)

    score = accuracy_score(yvl, y_pred_vl)
    print("accuracy_score", score)

    i+=1

    y_pred_test = model.predict(test)
    pred = model.predict_proba(xvl)[:, 1]

    sum_score += score

print("\nMean validation accuracy is:",sum_score/(i-1))

# %%
print(yvl)

# %%
print(y_pred_test)

# %% [markdown]
# Lets check the ROC Curve

# %%
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(yvl, pred, pos_label=1)

auc = metrics.roc_auc_score(yvl, pred)

plt.figure(figsize = (12,8))

plt.plot(fpr, tpr, label='validation, auc=' + str(auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend(loc=4)
plt.show()

# %%
submission['Loan_Status'] = y_pred_test
submission['Loan_ID'] = test_original['Loan_ID']
submission.head()

# %%
submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)
submission.head()

# %%
pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('logistic_kfold.csv', index=False)

# %% [markdown]
# ## 9. Based on domain knowledge, building new features which could affect the target variable

# %% [markdown]
# **1. Total Income: Applicant Income + Coapplicant Income**
# 
# **2. EMI: Ratio of Loan Amount with Loan Amount Term**
# 
# **3. Balance Income: Income left after paying EMI**

# %%
# train = train_original.copy()
# test = test_original.copy()

# %% [markdown]
# **1. Total Income**

# %%
train['TotalIncome'] = train.ApplicantIncome + train.CoapplicantIncome
test['TotalIncome'] = test.ApplicantIncome + test.CoapplicantIncome

# %%
sns.distplot(train.TotalIncome)

# %% [markdown]
# Shifted to the left, so right-skewed. So lets take the log transformation.

# %%
train['TotalIncome_log'] = np.log(train.TotalIncome)
test['TotalIncome_log'] = np.log(test.TotalIncome)

# %%
sns.distplot(train.TotalIncome_log)

# %%
sns.distplot(test.TotalIncome_log)

# %% [markdown]
# Now the distribution looks normal and effect of extreme values has been reduced.

# %% [markdown]
# **2. EMI**

# %%
train['EMI'] = train['LoanAmount']/train['Loan_Amount_Term']
test['EMI'] = test['LoanAmount']/test['Loan_Amount_Term']

# %%
sns.distplot(train.EMI)

# %%
sns.distplot(test.EMI)

# %% [markdown]
# **3. Balance Income**

# %%
train['BalanceIncome'] = train['TotalIncome'] - train['EMI']*1000
#Multiplying with 1000 to make the units equal

test['BalanceIncome'] = test['TotalIncome'] - test['EMI']*1000

# %%
sns.distplot(train['BalanceIncome'])

# %%
sns.distplot(test['BalanceIncome'])

# %% [markdown]
# **Now we need to drop the features using which we have created these new features, else there will be high correlation between the old features and the newly created features. Also, logistic regression assumes that the variables are not highly correlated. Moreover, noise has to be removed - so removing the correlated variables will reduce the noise also.**

# %%
train = train.drop(['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term'], axis=1)
test = test.drop(['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term'], axis=1)

# %% [markdown]
# ## 10. Model building with the newly created features

# %%
X = train.copy()
X.head()

# %%
X = X.drop(['Loan_Status_N', 'Loan_Status_Y'], axis=1)
X.head()

# %%
y.head()

# %%
sum_score = 0
i = 1

kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))

    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y.loc[train_index], y.loc[test_index]

    model = LogisticRegression(random_state=1)

    model.fit(xtr, ytr)

    y_pred_vl = model.predict(xvl)

    score = accuracy_score(yvl, y_pred_vl)
    print("accuracy_score", score)

    i+=1

    y_pred_test = model.predict(test)
    pred = model.predict_proba(xvl)[:, 1]

    sum_score += score

print("\nMean validation accuracy is:", sum_score/(i-1))

# %%
submission['Loan_Status'] = y_pred_test
submission['Loan_ID'] = test_original['Loan_ID']
submission.head()

# %%
submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)
submission.head()

# %%
pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('logistic_kfold_newfeatures.csv', index=False)

# %% [markdown]
# ### Using Decision Tree

# %%
from sklearn import tree

sum_score = 0
i = 1

kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))

    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    model = tree.DecisionTreeClassifier(random_state=1)

    model.fit(xtr, ytr)

    y_pred_vl = model.predict(xvl)

    score = accuracy_score(yvl, y_pred_vl)
    print("accuracy_score", score)

    i+=1

    y_pred_test = model.predict(test)
    pred = model.predict_proba(xvl)[:, 1]

    sum_score += score

print("\nMean validation accuracy is:", sum_score/(i-1))

# %%
submission['Loan_Status'] = y_pred_test
submission['Loan_ID'] = test_original['Loan_ID']

submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)
submission.head()

# %%
pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('decision_tree.csv', index=False)

# %% [markdown]
# ### Using Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

sum_score = 0
i = 1

kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))

    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    model = RandomForestClassifier(random_state=1)

    model.fit(xtr, ytr)

    y_pred_vl = model.predict(xvl)

    score = accuracy_score(yvl, y_pred_vl)
    print("accuracy_score", score)

    i+=1

    y_pred_test = model.predict(test)
    pred = model.predict_proba(xvl)[:, 1]

    sum_score += score

print("\nMean validation accuracy is:", sum_score/(i-1))

# %% [markdown]
# Mean Validation Accuracy for this model is 0.78. Now we can try to improve this by tuning the hyperparameters of the model - using GridSearch.

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

paramgrid = {'max_depth' : list(range(1, 20, 2)), 'n_estimators' : list(range(1, 200, 20))}

gridSearch = GridSearchCV(RandomForestClassifier(random_state=1), paramgrid)

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.3)

#Fit the grid search model
gridSearch.fit(X_train, y_train)

# %% [markdown]
# The optimized value for max_depth is 3 and n_estimators is 41. So, now we can build our model with these values.

# %%
sum_score = 0
i = 1

kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))

    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=121)

    model.fit(xtr, ytr)

    y_pred_vl = model.predict(xvl)

    score = accuracy_score(yvl, y_pred_vl)
    print("accuracy_score", score)

    i+=1

    y_pred_test = model.predict(test)
    pred = model.predict_proba(xvl)[:, 1]

    sum_score += score

print("\nMean validation accuracy is:", sum_score/(i-1))

# %%
submission['Loan_Status'] = y_pred_test
submission['Loan_ID'] = test_original['Loan_ID']

submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)
submission.head()

# %%
pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('random_forest.csv', index=False)

# %% [markdown]
# ### Lets find the feature importance now

# %%
importances = pd.Series(model.feature_importances_, index = X.columns)

importances

# %%
importances.plot(kind='barh', figsize=(12,8))

# %% [markdown]
# We can see that the most important feature is the Credit History followed by the new features we had created - Total Income, Balance Income and EMI.

# %% [markdown]
# ### XGBoost

# %%
from xgboost import XGBClassifier

sum_score = 0
i = 1

kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

y.replace('Y', 1, inplace=True)
y.replace('N', 0, inplace=True)

for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))

    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    model = XGBClassifier(max_depth=4, n_estimators=50)

    model.fit(xtr, ytr)

    y_pred_vl = model.predict(xvl)

    score = accuracy_score(yvl, y_pred_vl)
    print("accuracy_score", score)

    i+=1

    y_pred_test = model.predict(test)
    pred = model.predict_proba(xvl)[:, 1]

    sum_score += score

print("\nMean validation accuracy is:", sum_score/(i-1))

# %%
submission['Loan_Status'] = y_pred_test
submission['Loan_ID'] = test_original['Loan_ID']

submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)

pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('xgb.csv', index=False)

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

paramgrid = {'max_depth' : list(range(1, 20, 2)), 'n_estimators' : list(range(1, 200, 20))}

gridSearch = GridSearchCV(XGBClassifier(random_state=1), paramgrid)

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.3)

#Fit the grid search model
gridSearch.fit(X_train, y_train)

# %%
sum_score = 0
i = 1

kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

y.replace('Y', 1, inplace=True)
y.replace('N', 0, inplace=True)

for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))

    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    model = XGBClassifier(max_depth=1, n_estimators=41)

    model.fit(xtr, ytr)

    y_pred_vl = model.predict(xvl)

    score = accuracy_score(yvl, y_pred_vl)
    print("accuracy_score", score)

    i+=1

    y_pred_test = model.predict(test)
    pred = model.predict_proba(xvl)[:, 1]

    sum_score += score

print("\nMean validation accuracy is:", sum_score/(i-1))

# %%
submission['Loan_Status'] = y_pred_test
submission['Loan_ID'] = test_original['Loan_ID']

submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)

pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('xgb1.csv', index=False)


