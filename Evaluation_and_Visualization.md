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
  - Davies-Bouldin Index
  - Silhouette Score
# Regression Metrics
## R² (Coefficient of determination)
- **Definition:-** R² is a statistical measure that represents the proportion of variance for a dependent variable that's explained by an independent variable or variables in a regression model. It helps determine how well the model fits the data.
- **Formula**
```math
R² = 1 - \frac{Sum of Residual(SS_r)}{SS_t}

 = \frac{\sum (y_i-y'_i)^2}{\sum (y_i-y_m)^2}
```
- where
```math
SS_r =  \sum (y_i-y'_i)^2
```
  - Residual Sum of Squares measures the total deviation of the predicted values from the actual values.
```math
SS_t = \sum (y_i-y_m)^2
```
  - Total Sum of Squares measures the total deviation of the actual values from their mean.
-  **Interpretation** An R² value of 0 means the model explains none of the variability, while a value of 1 means it explains all variability. Higher values indicate a better fit of the model to the data.

## Adjusted R²
- **Definition:-** Adjusted R² modifies the R² value by penalizing the number of predictors in the model, helping to prevent overfitting.
- **Formula**
  ```math
  Adjusted R^2 = 1-\frac{(1-R^2)(n-1)}{n-k-1}
  ```
where n is the number of observations and k is the number of predictors(features)
-  **Interpretation** While R² can increase with more predictors, Adjusted R² only increases if the new predictor improves the model more than would be expected by chance. This makes it a more reliable metric when comparing models with different numbers of predictors.

## Mean Squared Error(MSE)
- **Definition:-** MSE quantifies the average of the squares of the errors—that is the average squared difference between the actual values and the values predicted by the model.
-  **Interpretation** Lower MSE values indicate a better fit, as they signify that the predicted values are close to the actual values. However, MSE is sensitive to outliers due to the squaring of errors.
- **Formula**
```math
MSE =  \frac{1}{n}\sum (y_i-y'_i)^2
```
## Mean Absolute Error(MAE)
- **Definition:-** MAE measures the average magnitude of errors in a set of predictions, without considering their direction (i.e., whether the prediction is over or under the actual value).
-  **Interpretation** MAE provides a straightforward interpretation: it gives the average error in the same units as the dependent variable. It is less sensitive to outliers than MSE, making it a robust metric for assessing model accuracy.
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


# Classification
## Confusion Matrix
- **Definition:-** A confusion matrix is a table used to evaluate the performance of a classification model. It provides a visual representation of the model's predictions compared to the actual outcomes.

| Actual\Prediction | Positive      | Negative      |
|-------------------|---------------|---------------|
|Positive           | True Positive | False Negative|
|Negative           | False Positive| True Negative |

  - True Positive (TP): The cases correctly predicted as positive.
  - False Negative (FN): The cases incorrectly predicted as negative (actual positive).
  - False Positive (FP): The cases incorrectly predicted as positive (actual negative).
  - True Negative (TN): The cases correctly predicted as negative.
-  **Interpretation** The confusion matrix allows you to see not just the overall accuracy of a model, but also the types of errors being made. It can be used to calculate various other metrics such as accuracy, precision, recall, and F1-score.
## Accuracy, Precision, Recall, F1 Score
### Accuracy
- **Definition:-** Accuracy measures the proportion of true results (both true positives and true negatives) among the total number of cases examined.
-  **Interpretation** Accuracy indicates the overall effectiveness of the model. However, it may not be reliable in imbalanced datasets, where one class significantly outnumbers the other.
```math
Acc=\frac{TP+TN}{TP+TN+FP+FN}
```
### Precision
- **Definition:-** Precision (also called Positive Predictive Value) measures the proportion of correct identifications.
-  **Interpretation** Precision is crucial in scenarios where the cost of false positives is high. For instance, in spam detection, high precision means fewer legitimate emails are misclassified as spam.
```math
p=\frac{TP}{TP+FP}
```
### Recall
- **Definition:-** Recall (also known as Sensitivity or True Positive Rate) measures the proportion of actual positives that were correctly identified.
-  **Interpretation** Recall is essential in cases where missing a positive case is critical, such as in disease detection. A high recall means that most actual positives are captured by the model
```math
r=\frac{TP}{TP+FN}
```
### F1 Score
- **Definition:-** The F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics.
-  **Interpretation** The F1 score is particularly useful when you need a single metric to compare models and when the classes are imbalanced. It takes both false positives and false negatives into account, making it a more comprehensive metric than accuracy alone.
```math
F1=\frac{2pr}{p+r}
```
## ROC-AUC 
- **Definition:-** The ROC curve is a graphical representation that illustrates the diagnostic ability of a binary classifier as its discrimination threshold is varied. The AUC (Area Under the Curve) quantifies the overall performance of the model.
- Components:
  - True Positive Rate (TPR): Equivalent to recall.
  - False Positive Rate (FPR): The proportion of negatives that are incorrectly identified as positives.
- **Interpretation:**
  - The ROC curve plots TPR against FPR at various threshold settings.
  - The AUC score ranges from 0 to 1:
    - AUC = 0.5 indicates no discrimination (random chance).
    - AUC = 1 indicates perfect discrimination.
  - A higher AUC value indicates a better model. The ROC-AUC metric is particularly useful for evaluating models in imbalanced datasets, as it considers both false positives and false negatives across all thresholds

# Clustering Metrics
## Davies-Bouldin Index
- **Definition:- **The Davies-Bouldin Index (DBI) is a metric for evaluating clustering algorithms. It considers both the intra-cluster distances and the inter-cluster distances to assess how well-separated and compact the clusters are.
- **Formula**
```math
DBI=\frac{1}{k}\sum_{i=1}^k max(i!=j)\frac{si+sj}{dij}
```
- Where:
  - k is the number of clusters.
  - si  is the average distance of points in cluster i to the centroid of cluster i (intra-cluster distance).
  - dij is the distance between the centroids of clusters i and j (inter-cluster distance).
- The lower the Davies-Bouldin Index, the better the clustering solution, as it indicates that clusters are compact and well-separated.
- A DBI of 0 would indicate perfect clustering; however, this is rarely achieved in practical applications.
- Generally, a lower DBI suggests better-defined clusters, while a higher DBI suggests more overlap between clusters.
## Silhouette Score
- **Definition:-** The Silhouette Score is a metric used to evaluate the quality of a clustering solution. It measures how similar an object is to its cluster compared to other clusters. The score provides insight into the appropriateness of the number of clusters chosen.
- **Formula**
```math
s(i)=\frac{b(i)-a(i)}{max(a(i), b(i))}
```
- where
- b(i) is the average distance between point i and all other points in the same cluster (intra-cluster distance).
- a(i) is the average distance between point i and all points in the nearest cluster (nearest-cluster distance).
- **Interpretation**
  - The Silhouette Score ranges from -1 to 1:
    - 1: The point is well-clustered, as it is closer to points in its cluster than to points in the nearest cluster.
    - 0: The point is on or very close to the decision boundary between two neighboring clusters.
    - -1: The point may have been assigned to the wrong cluster, as it is closer to points in another cluster.
- A higher average Silhouette Score across all points in the dataset indicates better-defined clusters.  

