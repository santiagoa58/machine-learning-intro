---
title: Linear Regression
description: Learn how to predict continuous values using linear relationships
prerequisites:
  - Basic Python programming
  - NumPy and Pandas fundamentals
  - Basic statistics (mean, variance)
  - Understanding of coordinate systems
learningOutcomes:
  - Build a simple linear regression model from scratch
  - Use scikit-learn for linear regression
  - Understand the cost function and gradient descent
  - Evaluate model performance with metrics
  - Apply linear regression to real-world problems
---

## Introduction

Linear regression is one of the most fundamental algorithms in machine learning and statistics. Despite its simplicity, it forms the foundation for understanding more complex algorithms and is still widely used in practice.

### What is Linear Regression?

Linear regression models the relationship between input variables (features) and a continuous output variable (target) by fitting a straight line (or hyperplane in higher dimensions) through the data points. The goal is to find the line that best describes the relationship between the variables.

### Real-World Applications

Linear regression is used extensively across industries:

- **Real Estate**: Predicting house prices based on square footage, number of bedrooms, location
- **Finance**: Forecasting stock prices, analyzing market trends
- **Healthcare**: Predicting patient outcomes, analyzing treatment effectiveness
- **Marketing**: Estimating sales based on advertising spend
- **Climate Science**: Modeling temperature trends, predicting weather patterns

### Learning Outcomes

By the end of this tutorial, you will:

1. Understand what linear regression is and when to use it
2. Build a simple linear regression model from scratch
3. Use scikit-learn's implementation for real-world problems
4. Understand the mathematics behind the algorithm
5. Evaluate and improve your models

---

## Section 1: Building Your First Model

Let's dive right in and build a linear regression model. We'll start with a simple example to understand the core concepts.

### The Problem: Predicting Student Exam Scores

Imagine you're a teacher who wants to predict students' final exam scores based on the number of hours they studied. You have data from previous students:

| Hours Studied | Exam Score |
|---------------|------------|
| 1             | 55         |
| 2             | 65         |
| 3             | 70         |
| 4             | 75         |
| 5             | 85         |

### Step 1: Loading and Exploring Data

First, let's load this data and visualize it:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Our dataset
hours_studied = np.array([1, 2, 3, 4, 5])
exam_scores = np.array([55, 65, 70, 75, 85])

# Create a DataFrame for better visualization
data = pd.DataFrame({
    'Hours Studied': hours_studied,
    'Exam Score': exam_scores
})

print(data)

# Visualize the relationship
plt.figure(figsize=(8, 6))
plt.scatter(hours_studied, exam_scores, color='blue', s=100)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Hours Studied vs Exam Score')
plt.grid(True, alpha=0.3)
plt.show()
```

**What do you notice?** There appears to be a positive relationship: as study hours increase, exam scores tend to increase. This is a good candidate for linear regression!

### Step 2: Building the Model with scikit-learn

Let's use scikit-learn to build our first linear regression model:

```python
from sklearn.linear_model import LinearRegression

# Reshape data for sklearn (needs 2D arrays)
X = hours_studied.reshape(-1, 1)  # Features (hours studied)
y = exam_scores                    # Target (exam scores)

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

print(f"Model Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")
```

### Step 3: Understanding the Results

The model learns two key parameters:
- **Slope (coefficient)**: How much the exam score changes for each additional hour of study
- **Intercept**: The predicted score if someone studied 0 hours

Our linear equation becomes:
```
Exam Score = (slope × Hours Studied) + intercept
```

### Step 4: Making Predictions

Now we can predict scores for new students:

```python
# Predict score for a student who studied 6 hours
new_hours = np.array([[6]])
predicted_score = model.predict(new_hours)

print(f"Predicted score for 6 hours of study: {predicted_score[0]:.2f}")

# Visualize the fitted line
plt.figure(figsize=(8, 6))
plt.scatter(hours_studied, exam_scores, color='blue', s=100, label='Actual Data')
plt.plot(hours_studied, predictions, color='red', linewidth=2, label='Fitted Line')
plt.scatter(6, predicted_score, color='green', s=150, marker='*',
            label='Prediction for 6 hours', zorder=5)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Linear Regression Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Step 5: Evaluating the Model

How good is our model? Let's measure its performance:

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Calculate metrics
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.3f}")
```

**Understanding the Metrics:**
- **MSE/RMSE**: Average squared/root squared difference between predictions and actual values (lower is better)
- **MAE**: Average absolute difference (more interpretable than MSE)
- **R²**: Proportion of variance explained by the model (0 to 1, higher is better)

---

## Section 2: Real-World Application

Now that you understand the basics, let's apply linear regression to a more realistic problem: predicting house prices.

### The Dataset

We'll use a dataset with multiple features:
- Square footage
- Number of bedrooms
- Number of bathrooms
- Age of the house

### Multi-Feature Linear Regression

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample house data
data = {
    'sqft': [1500, 1800, 2400, 2000, 3000, 1200, 2200, 2600],
    'bedrooms': [3, 3, 4, 3, 5, 2, 4, 4],
    'bathrooms': [2, 2, 3, 2.5, 3, 1, 2.5, 3],
    'age': [10, 5, 2, 8, 1, 20, 4, 3],
    'price': [250000, 280000, 350000, 290000, 450000, 180000, 320000, 380000]
}

df = pd.DataFrame(data)

# Separate features and target
X = df[['sqft', 'bedrooms', 'bathrooms', 'age']]
y = df['price']

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

# Feature importance (coefficients)
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: ${coef:,.2f}")
```

### Best Practices

1. **Always split your data**: Use separate training and testing sets to evaluate performance
2. **Scale your features**: Standardization helps when features have different units
3. **Check for linearity**: Linear regression assumes a linear relationship
4. **Watch for outliers**: They can significantly impact your model
5. **Validate assumptions**: Check residuals for patterns

---

## Section 3: The Mathematics Behind Linear Regression

Now that you've built models, let's understand the mathematics that makes linear regression work.

### The Linear Model

For a simple linear regression with one feature:
```
y = β₀ + β₁x
```

Where:
- `y` is the predicted value
- `x` is the input feature
- `β₁` is the slope (coefficient)
- `β₀` is the intercept

For multiple features:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### The Cost Function

How do we find the best line? We minimize the **Mean Squared Error (MSE)**:

```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

Where:
- `yᵢ` is the actual value
- `ŷᵢ` is the predicted value
- `n` is the number of samples

This is also called the **cost function** or **loss function**. The goal is to find parameters (β₀, β₁, ...) that minimize this cost.

### Finding the Minimum: Two Approaches

#### 1. Normal Equation (Closed-form Solution)

For simple problems, we can calculate the optimal parameters directly:

```
β = (XᵀX)⁻¹Xᵀy
```

This gives us the exact solution in one step, but becomes computationally expensive for large datasets.

#### 2. Gradient Descent (Iterative Solution)

For larger problems, we use **gradient descent**:

1. Start with random parameters
2. Calculate the gradient (direction of steepest increase)
3. Take a step in the opposite direction
4. Repeat until convergence

```python
# Simple gradient descent implementation
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])

    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient

    return theta

# Add intercept term
X_with_intercept = np.c_[np.ones(len(X)), X]

# Train using gradient descent
theta = gradient_descent(X_with_intercept, y)
print(f"Learned parameters: {theta}")
```

### Assumptions of Linear Regression

Linear regression makes several important assumptions:

1. **Linearity**: The relationship between X and y is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Features are not highly correlated

**Important:** Always check these assumptions! If violated, your model may perform poorly or give misleading results.

---

## Section 4: Conclusion

### What You've Learned

Congratulations! You've now mastered the fundamentals of linear regression:

✅ Built linear regression models from scratch
✅ Used scikit-learn for real-world applications
✅ Understood evaluation metrics (MSE, RMSE, R²)
✅ Learned the mathematical foundation
✅ Discovered when and how to apply linear regression

### Key Takeaways

1. **Start simple**: Linear regression is a great baseline model
2. **Check assumptions**: Ensure your data meets the requirements
3. **Evaluate properly**: Use train/test splits and appropriate metrics
4. **Interpret results**: Understand what the coefficients mean
5. **Know the limitations**: Linear regression works best for linear relationships

### When to Use Linear Regression

**Use linear regression when:**
- You need to predict continuous values
- The relationship between variables is roughly linear
- You need an interpretable model
- You have relatively clean data without too many outliers

**Consider alternatives when:**
- The relationship is highly non-linear (try polynomial regression or other algorithms)
- You're doing classification (try logistic regression)
- You have many outliers (try robust regression methods)
- Features are highly correlated (try regularization: Ridge or Lasso)

### Next Steps

Now that you understand linear regression, you're ready to explore:
- **Polynomial Regression**: For non-linear relationships
- **Ridge and Lasso Regression**: For regularization and feature selection
- **Logistic Regression**: For classification problems
- **K-Nearest Neighbors**: For non-parametric learning

Keep practicing, and remember: linear regression is a powerful tool that forms the foundation for many advanced machine learning techniques!
