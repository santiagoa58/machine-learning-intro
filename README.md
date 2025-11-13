# Machine Learning Introduction

A comprehensive, hands-on introduction to machine learning that prioritizes understanding through application. This project teaches ML concepts by starting with real-world examples and working backwards to the theory, making complex topics accessible and engaging.

## Philosophy

**Learn by Doing, Understand by Exploring**

This isn't your typical ML tutorial that starts with pages of mathematics and theory. Instead, we believe the best way to learn machine learning is to:

1. **See it work first** - Start with a real application and get results
2. **Understand what it does** - Explore the behavior and capabilities
3. **Learn why it works** - Dive into the theory with context and motivation
4. **Master the details** - Deep dive into the mathematics and implementation

Every tutorial follows the **"From Application to Theory"** approach, ensuring you always understand *why* you're learning something before diving into the *how*.

## What You'll Learn

### Supervised Learning

**Regression Algorithms:**
- **Linear Regression** - Predict continuous values (e.g., stock prices)
  - Basic implementation with real stock market data
  - Deep dive into optimization, gradient descent, and mathematical foundations

**Classification Algorithms:**
- **K-Nearest Neighbors (KNN)** - Classify based on proximity to training examples
  - Multi-class classification with the Iris dataset
  - Distance metrics, curse of dimensionality, and performance optimization

- **Logistic Regression** - Binary classification with probabilistic outputs
  - Sentiment analysis on movie reviews
  - Sigmoid functions, log loss, and maximum likelihood estimation

- **Support Vector Machines (SVM)** - Find optimal decision boundaries
  - Face similarity detection using the LFW dataset
  - Kernel methods, PCA dimensionality reduction, and hyperparameter tuning

### Foundations

Essential tools and libraries for machine learning:
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis
- **scikit-learn** - ML algorithms and utilities

## Project Structure

```
machine-learning-intro/
├── foundations/
│   └── programming-and-tools/
│       ├── numpy.ipynb
│       ├── pandas.ipynb
│       └── sklearn.ipynb
├── supervised-learning/
│   ├── linear-regression/
│   │   ├── Linear Regression.ipynb
│   │   └── Linear Regression - Deeper Dive.ipynb
│   ├── logistic-regression/
│   │   └── Logistic Regression.ipynb
│   ├── k-nearest-neighbor/
│   │   └── knn.ipynb
│   └── support-vector-machines/
│       └── svm.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Basic understanding of Python programming
- Familiarity with basic algebra (calculus helpful but not required)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/santiagoa58/machine-learning-intro.git
cd machine-learning-intro
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start learning! Open any notebook in Jupyter:
```bash
jupyter notebook
```

### Recommended Learning Path

**For Complete Beginners:**
1. Start with `foundations/programming-and-tools/` to get familiar with the essential libraries
2. Begin with `Linear Regression` - it's the foundation of many ML algorithms
3. Move to `K-Nearest Neighbors` - simple yet powerful classification
4. Progress to `Logistic Regression` - introduces probabilistic thinking
5. Explore `Support Vector Machines` - more advanced classification

**For Those with ML Background:**
- Jump directly to any algorithm that interests you
- Each tutorial is self-contained with links to prerequisites

## Key Features

### Application-First Approach
Every tutorial starts with a real-world problem:
- Predicting stock prices
- Classifying flower species
- Analyzing movie review sentiment
- Detecting face similarity

### Progressive Complexity
- Start simple, build up gradually
- Code examples before mathematical formulas
- Theory presented only after seeing it work

### Comprehensive Explanations
- Not overly dry or academic
- Plain language with minimal jargon
- Every technical term is explained when introduced
- Visual aids and plots throughout

### Absolute Correctness
- All mathematical formulations are precise
- Code examples are tested and verified
- Common misconceptions are explicitly addressed

## Contributing

This is a learning project focused on educational clarity and technical accuracy. If you find any errors or have suggestions for improvements, please open an issue or submit a pull request.

Please review [PROJECT_GUIDELINES.md](PROJECT_GUIDELINES.md) before contributing to ensure your additions follow the established teaching philosophy.

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Built using [scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), and [Matplotlib](https://matplotlib.org/)
- Datasets from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) and [scikit-learn datasets](https://scikit-learn.org/stable/datasets.html)
- Inspired by the belief that machine learning should be accessible to everyone

---

**Start your ML journey today - no PhD required!**
