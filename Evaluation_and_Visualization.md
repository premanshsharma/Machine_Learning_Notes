# Mind Map
- Regression Metrics
  - R², Adjusted R²
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Bias vs. Variance
- Classification Metrics
  - Confusion Matrix
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC
- Clustering Metrics
  - Silhouette Score
  - Davies-Bouldin Index
# Regression Metrics
## R² (Coefficient of determination)
**Formula**
```math
R² = 1 - \frac{Sum of Residual(SS_res)}{SS_total}

R² = \frac{\sum (y_i-y'_i)^2}{\sum (y_i-y'_mean)^2}
```
## Advantages and Disadvantages
| Metric                 | Advantages                                          | Disadvantages                                      |
|-----------------------|----------------------------------------------------|---------------------------------------------------|
| R²                    | - Easy to interpret                                 | - Can be misleading for non-linear relationships   |
|                       | - Useful for comparing models                       | - Not a definitive measure of model performance    |
| Adjusted R²           | - Adjusts for the number of predictors             | - Still susceptible to misleading interpretations   |
|                       | - Helps in model selection                          | - Not robust to outliers                           |
| Mean Squared Error    | - Sensitive to large errors                         | - Can be difficult to interpret due to squaring    |
|                       | - Useful for optimization of models                 | - Gives more weight to outliers                    |
| Mean Absolute Error    | - Easy to interpret                                 | - Less sensitive to large errors                    |
|                       | - Useful for assessing model accuracy               | - Can be misleading if the data has outliers      |
| Bias vs. Variance     | - Helps diagnose model performance                  | - Requires careful balancing in model training     |
|                       | - Indicates underfitting vs. overfitting           | - No quantitative measure alone for model quality   |
