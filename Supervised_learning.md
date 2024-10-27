# Mindmap
- Distance-Based Methods
  - K-Nearest Neighbors (KNN)
- Probability-Based Methods
  - Naive Bayes Classification
  - Maximum Likelihood Estimation (MLE)
  - Maximum A Posteriori Estimation (MAP)
- Line-Based Methods
  - Linear Regression
  - Support Vector Machines (SVM)
  - Logistic Regression
- Tree-Based and Ensembling Methods
  - Decision Trees
  - Ensembling Techniques
    - Bagging
      - Random Forest
    - Boosting
      - AdaBoost
      - XGBoost
# Probability-Based Methods
## 1. Maximum Likelihood Estimation (MLE)
- MLE is a method to estimate the parameters of a probabilistic model by maximizing the likelihood of observing the data given the model.
- The idea is to find the parameter values that make the observed data most likely.
- It finds the best-fit parameters for a model that explain the data we have.
### Algorithm
- Step 1: Define the Model
  - Data follows a known distribution, let's say it is a Gaussian(Normal) distribution
  - Mean μ (average of the data)
  - Standard deviation σ (spread of the data)
  - Probability density function (PDF) f(x|μ,σ) is
```math
f(x|μ,σ) = \frac{1}{σ\sqrt{2pi}}e^(\frac{-(x-μ)^2}{2σ^2})
```
  - This function tells us how likely a data point x is, given parameters μ and σ
- Step 2: Write the likelihood Function
  - likelihood function is the probability of observing all the data points, given the parameters μ and σ.
  - L(μ, σ|x1, x2, ........, xn) = f(x1|μ,σ) x f(x2|μ,σ) ........ x f(xn|μ,σ)
  - L(μ, σ|x1, x2, ........, xn) = ∏f(xi|μ,σ)
  - This function tells the goodness of the parameters μ and σ in explaining the data.
- Step 3: Log Likelihood
  - Taking product is hard so we take log
    -  log(L(μ, σ|x1, x2, ........, xn)) = ∑<sup>n</sup> log(f(xi|μ,σ))
    -  For normal distribution it becomes
      -![image](https://github.com/user-attachments/assets/09c22cf1-921c-4da7-92a0-16b00d7f379f)
- Step 4: Maximize the log-likelihood
  - Now, we want to find the values of μ and σ that maximize the log-likelihood. In mathematical terms, we need to solve for the parameters by finding the derivatives of the log-likelihood with respect to μ and σ, and setting them to zero (this is how you find the maximum in calculus).
   - ![image](https://github.com/user-attachments/assets/bf0a1a83-4544-4fdf-8238-3e548851a17f)
- Step 5: Interpretation
  - The value of μ that maximizes the likelihood is the sample mean.
  - The value of σ that maximizes the likelihood is the sample standard deviation.
### General MLE Approach
- Write the probability (or likelihood) of the data given the model parameters.
- Take the product of probabilities for all data points to get the likelihood function.
- Take the logarithm to get the log-likelihood.
- Maximize the log-likelihood by taking derivatives to the parameters and solving.
