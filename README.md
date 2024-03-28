# Fraud detection project

A project to determine fraud cases in card payment transactions. Only an extremely tiny percentage of cases are fraudulent, precisely 0.74% of transactions. Moreover, the bank possesses resources to follow up on only 400 possible fraudulent cases per month. I've added a 'fraudCase' column as target feature, with label 1 for fraud cases and 0 otherwise.

## Data cleaning

'merchantZip' contains 3260 unique categories, 19.4% of which are missing values. That figure increases to 31.6% if you include all of the entries marked as '0'. The remaining, unique 'merchantZip' codes all figure below 1%, so also due to the high fragmentation I decided to drop this feature.

I decided to convert 'mcc', 'merchantCountry', 'posEntryMode' to categorical features, and calculate the [mutual information score from Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score) for each categorical feature against the fraud case target. I also calucalted the Pearson correlation on the numerical features.
