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
- R² measures the proportion of variance in the dependent variable that is predictable from the independent variable(s).
**Formula**
```math
R² = 1 - \frac{Sum of Residual(SS_r)}{SS_t}

 = \frac{\sum (y_i-y'_i)^2}{\sum (y_i-y_m)^2}
```
```math
SS_r =  \sum (y_i-y'_i)^2
```
Residual Sum of Squares measures the total deviation of the predicted values from the actual values.
```math
SS_t = \sum (y_i-y_m)^2
```
Total Sum of Squares measures the total deviation of the actual values from their mean.
- An R² value of 0 means the model explains none of the variability, while a value of 1 means it explains all variability. Higher values indicate a better fit of the model to the data.

## Adjusted R²
- Adjusted R² modifies the R² value by penalizing the number of predictors in the model, helping to prevent overfitting.
- **Formula**
  ```math
  Adjusted R^2 = 1-\frac{(1-R^2)(n-1)}{n-k-1}
  ```
where n is the number of observations and k is the number of predictors(features)
- While R² can increase with more predictors, Adjusted R² only increases if the new predictor improves the model more than would be expected by chance. This makes it a more reliable metric when comparing models with different numbers of predictors.

## Mean Squared Error(MSE)
- MSE quantifies the average of the squares of the errors—that is the average squared difference between the actual values and the values predicted by the model.
- Lower MSE values indicate a better fit, as they signify that the predicted values are close to the actual values. However, MSE is sensitive to outliers due to the squaring of errors.
- **Formula**
```math
MSE =  \frac{1}{n}\sum (y_i-y'_i)^2
```
## Mean Absolute Error(MAE)
- MAE measures the average magnitude of errors in a set of predictions, without considering their direction (i.e., whether the prediction is over or under the actual value).
- MAE provides a straightforward interpretation: it gives the average error in the same units as the dependent variable. It is less sensitive to outliers than MSE, making it a robust metric for assessing model accuracy.
- **Formula**
```math
MAE =  \frac{1}{n}\sum |y_i-y'_i|
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
