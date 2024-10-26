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
| Metric                 | Formula                                          | Interpretation                                         | Advantages                                          | Disadvantages                                      |
|-----------------------|--------------------------------------------------|-------------------------------------------------------|----------------------------------------------------|---------------------------------------------------|
| R²                    | \( R^2 = 1 - \frac{SS_{res}}{SS_{tot}} \)      | Proportion of variance explained by the model         | - Easy to interpret                                 | - Can be misleading for non-linear relationships   |
|                       |                                                  |                                                       | - Useful for comparing models                       | - Not a definitive measure of model performance    |
| Adjusted R²           | \( \text{Adjusted } R^2 = 1 - \left( \frac{(1-R^2)(n-1)}{n-k-1} \right) \) | Adjusts for the number of predictors                   | - Adjusts for the number of predictors             | - Still susceptible to misleading interpretations   |
|                       |                                                  |                                                       | - Helps in model selection                          | - Not robust to outliers                           |
| Mean Squared Error    | \( \text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2 \) | Average of the squared errors                           | - Sensitive to large errors                         | - Can be difficult to interpret due to squaring    |
|                       |                                                  |                                                       | - Useful for optimization of models                 | - Gives more weight to outliers                    |
| Mean Absolute Error    | \( \text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i| \)  | Average magnitude of errors                             | - Easy to interpret                                 | - Less sensitive to large errors                    |
|                       |                                                  |                                                       | - Useful for assessing model accuracy               | - Can be misleading if the data has outliers      |
| Bias vs. Variance     | N/A                                              | Indicates underfitting vs. overfitting                | - Helps diagnose model performance                  | - Requires careful balancing in model training     |
|                       |                                                  |                                                       | - Indicates model performance issues                 | - No quantitative measure alone for model quality   |
