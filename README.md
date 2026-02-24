# Credit card fraud in payment transactions

## Introduction

This is a machine learning project for determining fraud cases in card payment transactions. More specifically, it is a binary classification problem where the target feature is the 'fraudCase' column, with label 1 for fraud cases and 0 otherwise. Only an extremely tiny percentage of cases are fraudulent, precisely 0.74% of all transactions.

## Data preparation

'merchantZip' contains 3260 unique categories, 19.4% of which are missing values. That figure increases to 31.6% if you include all of the entries marked as '0'. The remaining, unique 'merchantZip' codes all figure below 1%, so also due to the high fragmentation I decided to drop this feature.

In the analysis I decided to convert 'eventId', 'accountNumber', 'merchantId', 'mcc', 'merchantCountry', 'merchantZip', 'posEntryMode' to string data type, because 'DictVectorizer' will only do a binary one-hot encoding when feature values are of type string. However, 'eventId' and 'merchantId' are dropped from the dataset due to the high amount of unique values which doesn't offer any discrimination. 'transactionTime' is set as a datetime type but also dropped from the dataset. 'transactionAmount' and 'availableCash' are the only two numerical data types and are kept.

## Exploratory data analysis

The exploratory data analysis plots show the frequency of some fraudulent entries of the category features that have suspicious behaviors and that warrant more attention. For example, many of the fraudulent transactions are small in amount. The frequency of one fraudulent merchant code (5735) is particularly high (almost 120 out of 875 fradulent transactions). In particular, the merchant account '8b9c15ea' stands out as one account aimed by fraudsters.

Three fraudulent country codes particularly stand out (826, 840, 442), while the merchant code most targeted by fraudsters is '0'. However, this may not be that informative as many ZIP codes in the US start with '0'.

Just from exploratory data analysis one can see that fraudulent POS entry codes are most likely to be '1' and '81', which are respectively 'POS Entry Mode Manual' and the 'POS Entry E-Commerce' mode. This last mode has many fraudulent cases, and deserves special attention for the possibility of fraud.

On the string data types, I calculate the [mutual information score from Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score) against the fraud case target. The mutual information score is also called information gain. I also calcualted the Pearson correlation for 'transactionAmount' and 'availableCash', but there is no correlation between the two numerical features.

## Modeling

I used the 'RandomUnderSampler' class from imbalanced-learn to under-sample the majority class. Since in my dataset 875 cases are fradulent, the imbalanced-learn class randomly selects 875 non-fraud cases to generate a balanced dataset. I used the the balanced dataset to build three, simple maching learning classifiers: logistic regression, decision tree classifier and random forest. Of all three the logistic regression obtained the best balanced accuracy score, from 5-fold cross validation, at 0.894 +/- 0.005 on the **unbalanced** test dataset. The balanced accuracy score is defined as the mean of the recall of the two target classes.
