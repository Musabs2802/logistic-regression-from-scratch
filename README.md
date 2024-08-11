# Logistic Regression from Scratch

## What is Logistic Regression?
Logistic Regression is a statistical method used for binary classification problems, where the output can take one of two possible values (e.g., yes/no, 0/1, true/false). It models the probability that a given input belongs to a particular category by fitting a logistic function (also known as the sigmoid function) to the input features. Logistic Regression estimates the probability of an event occurring based on the given input variables.

## Properties of Logistic Regression
- **Probabilistic Interpretation**: Outputs a probability value between 0 and 1.
- **Linear Decision Boundary**: The decision boundary is linear in the feature space.
- **Non-linear Mapping**: The logistic function maps linear combinations of inputs to a probability.
- **Interpretable Coefficients**: Coefficients indicate the strength and direction of the relationship between features and the target.

## Assumptions of Logistic Regression
1. **Linearity of Independent Variables and Log Odds**: The relationship between the independent variables and the log odds of the dependent variable is linear.
2. **Independence of Errors**: The observations should be independent of each other.
3. **No Multicollinearity**: Independent variables should not be highly correlated with each other.
4. **Large Sample Size**: Logistic regression requires a large sample size to provide reliable estimates.

## Equation of Logistic Regression
The logistic regression model predicts the probability $$\ P \$$ that a given input $$\ \mathbf{x} \$$ belongs to the positive class (1) using the logistic function:

$$\
P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
\$$

Where:
- $$\ \sigma \$$ is the logistic (sigmoid) function.
- $$\ \mathbf{w} \$$ is the vector of weights.
- $$\ \mathbf{x} \$$ is the input feature vector.
- $$\ b \$$ is the bias term.

The logit function (log-odds) is given by:

$$\
\text{logit}(P) = \ln\left(\frac{P}{1-P}\right) = \mathbf{w}^T \mathbf{x} + b
\$$

## How to Calculate Logistic Regression

### Step 1: Initialize Parameters
Initialize the weights $$\ \mathbf{w} \$$ and bias $$\ b \$$ to small random values.

### Step 2: Compute the Predicted Probability
For each input $$\ \mathbf{x}^{(i)} \$$, compute the predicted probability $$\ \hat{y}^{(i)} \$$ using the logistic function:

$$\
\hat{y}^{(i)} = \sigma(\mathbf{w}^T \mathbf{x}^{(i)} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x}^{(i)} + b)}}
\$$

### Step 3: Compute the Loss Function
Compute the loss function using the binary cross-entropy loss:

$$\
L(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]
\$$

### Step 4: Compute the Gradients
Compute the gradients of the loss function with respect to the weights $$\ \mathbf{w} \$$ and bias $$\ b \$$:

$$\
\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) \mathbf{x}^{(i)}
\$$

$$\
\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})
\$$

### Step 5: Update Parameters
Update the weights $$\ \mathbf{w} \$$ and bias $$\ b \$$ using gradient descent:

$$\
\mathbf{w} := \mathbf{w} - \alpha \frac{\partial L}{\partial \mathbf{w}}
\$$

$$\
b := b - \alpha \frac{\partial L}{\partial b}
\$$

where $$\ \alpha \$$ is the learning rate.

### Step 6: Repeat
Repeat steps 2 to 5 until the loss function converges or a predefined number of iterations is reached.


## When to Use Logistic Regression

1. **Binary Outcomes**: When the dependent variable is binary (i.e., it has two possible outcomes like Yes/No, True/False, 0/1).
2. **Linearly Separable Data**: When the data can be linearly separated, meaning a straight line (or hyperplane in higher dimensions) can be used to separate the two classes.
3. **Probability Prediction**: When you need not only the classification outcome but also the probability of a particular class.
4. **Simple and Fast**: When you need a quick and easy-to-implement model that works well for a baseline or when computational resources are limited.
5. **Interpretability**: When model interpretability is important, as logistic regression coefficients can provide insights into the relationship between the predictor variables and the probability of the outcome.

## Advantages of Logistic Regression

1. **Simplicity**: Logistic Regression is straightforward to implement and understand, making it a good baseline model.
2. **Efficiency**: It requires less computational power and can handle large datasets efficiently.
3. **Probability Interpretation**: It provides probability scores for observations, which can be useful in various decision-making processes.
4. **Feature Importance**: The coefficients of the model can be used to understand the influence of different features on the outcome.
5. **Regularization**: Techniques like L1 (Lasso) and L2 (Ridge) regularization can be easily incorporated to prevent overfitting.

## Disadvantages of Logistic Regression

1. **Linear Decision Boundary**: Logistic Regression assumes a linear relationship between the independent variables and the log odds of the outcome, which might not capture complex patterns.
2. **Not Suitable for Non-linear Problems**: For non-linear classification problems, logistic regression may not perform well without transformations or feature engineering.
3. **Sensitive to Outliers**: The performance can be affected by the presence of outliers.
4. **Overfitting with High-Dimensional Data**: When the number of features is very large, logistic regression might overfit the training data, especially if regularization is not applied.
5. **Binary Limitation**: Standard logistic regression is limited to binary classification. For multi-class classification, extensions like multinomial logistic regression or other algorithms are needed.

