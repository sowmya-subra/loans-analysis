# Stat 451 Final Project: Loans Analysis

## Introduction

This project analyzed a loans dataset sourced from LendingClub.com to predict loan defaults. The primary objectives were:

1. To classify whether a borrower will repay their loan based on various feature variables.
2. To identify which features are most important in predicting loan repayment.

The goal was to achieve a classification accuracy score greater than the proportion of loans being repaid, which was 84%. Initial attempts with a decision tree model showed significant overfitting. Further exploration of models revealed that DecisionTree, LogisticRegression, and RandomForest provided the best accuracy. However, the analysis of feature importance indicated that none of the features were highly effective in predicting loan defaults.

## Data Source and Size

- **Source**: Kaggle
- **Size**: 10,000 rows
- **Timeframe**: 2007 to 2010

The dataset included variables such as interest rate, monthly installment, debt-to-income ratio, FICO score, unpaid amount, delinquency frequency, loan purpose, and log annual income. The focus was on distinguishing between loans that were repaid (not.fully.paid = 0) and those that were not (not.fully.paid = 1).

## Data Exploration

- **Income**: Comparative graphs showed similar median log annual income values for both paid and unpaid loans.
- **Delinquencies**: Most borrowers had zero delinquencies, with the repayment ratio approximately proportional across delinquency values.
- **Repayment Ratio**: With 84% of loans repaid, predicting all loans as repaid would result in an accuracy score of 84%.

## Feature Engineering

- **One-Hot Encoding**: Applied to the loan purpose variable.
- **Normalization**: Min-max normalization was applied to all variables.
- **Feature Selection**: Techniques such as SelectKBest, f_classif, and VarianceThreshold were tested but ultimately discarded due to lower accuracy scores.

## Model Comparison

- **Initial Model**: The simple decision tree model exhibited overfitting, with a training accuracy of 1 and a validation accuracy of 0.7537.
- **Hyperparameter Tuning**: GridSearchCV was used to optimize classifiers: LogisticRegression, DecisionTree, KNN, and RandomForest.
  - **Logistic Regression**: C = 0.01, 10, 1000
  - **Decision Tree Classifier**: criterion = 'entropy', max_depth = 1
  - **Random Forest Classifier**: max_depth = 11

Testing revealed that all models achieved an accuracy of 0.84. The DecisionTree (depth 1) and LogisticRegression models predicted no loan defaults, while the RandomForest model predicted one false positive.

## Feature Importance

- **Lasso Regression**: Applied with C = 1000 and a 0.1 threshold. The distributions of key variables (delinq.2.yrs and log.annual.inc) for repaid versus unpaid loans were similar, raising concerns about data validity.
- **Interest Rate**: Despite having the highest coefficient value, the interest rate did not significantly improve predictions. The LogisticRegression model predicted no defaults due to the absence of high interest rates in the dataset.
- **Permutation Importance**: Addressing multicollinearity through hierarchical clustering on Spearman correlations reduced the variables to 3, with minimal impact on accuracy.

## Conclusion

The analysis did not yield a conclusive model for predicting loan defaults or effectively classifying borrowers based on the available features. Predicting no defaults remains the most effective approach until more informative variables are identified. Future work could involve exploring additional features, alternative evaluation metrics like the F1 score, and potential improvements in feature engineering and model selection.

## Future Work

- Explore additional features or external data sources.
- Consider alternative evaluation metrics such as the F1 score.
- Investigate potential improvements in feature engineering and model selection.

## Acknowledgments

Special thanks to LendingClub.com for providing the dataset and Kaggle for hosting it.
