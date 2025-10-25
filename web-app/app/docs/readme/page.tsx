export default function ReadmePage() {
  return (
    <>
      <h1>Machine Learning Introduction</h1>

      <p>
        A comprehensive, hands-on introduction to machine learning that prioritizes understanding
        through application. This project teaches ML concepts by starting with real-world examples
        and working backwards to the theory, making complex topics accessible and engaging.
      </p>

      <h2>Philosophy</h2>

      <p><strong>Learn by Doing, Understand by Exploring</strong></p>

      <p>
        This isn't your typical ML tutorial that starts with pages of mathematics and theory.
        Instead, we believe the best way to learn machine learning is to:
      </p>

      <ol>
        <li><strong>See it work first</strong> - Start with a real application and get results</li>
        <li><strong>Understand what it does</strong> - Explore the behavior and capabilities</li>
        <li><strong>Learn why it works</strong> - Dive into the theory with context and motivation</li>
        <li><strong>Master the details</strong> - Deep dive into the mathematics and implementation</li>
      </ol>

      <p>
        Every tutorial follows the <strong>"From Application to Theory"</strong> approach, ensuring
        you always understand <em>why</em> you're learning something before diving into the <em>how</em>.
      </p>

      <h2>What You'll Learn</h2>

      <h3>Supervised Learning</h3>

      <p><strong>Regression Algorithms:</strong></p>
      <ul>
        <li>
          <strong>Linear Regression</strong> - Predict continuous values (e.g., stock prices)
          <ul>
            <li>Basic implementation with real stock market data</li>
            <li>Deep dive into optimization, gradient descent, and mathematical foundations</li>
          </ul>
        </li>
      </ul>

      <p><strong>Classification Algorithms:</strong></p>
      <ul>
        <li>
          <strong>K-Nearest Neighbors (KNN)</strong> - Classify based on proximity to training examples
          <ul>
            <li>Multi-class classification with the Iris dataset</li>
            <li>Distance metrics, curse of dimensionality, and performance optimization</li>
          </ul>
        </li>
        <li>
          <strong>Logistic Regression</strong> - Binary classification with probabilistic outputs
          <ul>
            <li>Sentiment analysis on movie reviews</li>
            <li>Sigmoid functions, log loss, and maximum likelihood estimation</li>
          </ul>
        </li>
        <li>
          <strong>Support Vector Machines (SVM)</strong> - Find optimal decision boundaries
          <ul>
            <li>Face similarity detection using the LFW dataset</li>
            <li>Kernel methods, PCA dimensionality reduction, and hyperparameter tuning</li>
          </ul>
        </li>
      </ul>

      <h3>Foundations</h3>

      <p>Essential tools and libraries for machine learning:</p>
      <ul>
        <li><strong>NumPy</strong> - Numerical computing and array operations</li>
        <li><strong>Pandas</strong> - Data manipulation and analysis</li>
        <li><strong>scikit-learn</strong> - ML algorithms and utilities</li>
      </ul>

      <h2>Project Structure</h2>

      <pre><code>{`machine-learning-intro/
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
└── README.md`}</code></pre>

      <h2>Getting Started</h2>

      <h3>Prerequisites</h3>

      <ul>
        <li>Python 3.7 or higher</li>
        <li>Basic understanding of Python programming</li>
        <li>Familiarity with basic algebra (calculus helpful but not required)</li>
      </ul>

      <h3>Installation</h3>

      <ol>
        <li>
          Clone the repository:
          <pre><code>{`git clone https://github.com/santiagoa58/machine-learning-intro.git
cd machine-learning-intro`}</code></pre>
        </li>
        <li>
          Install dependencies:
          <pre><code>pip install -r requirements.txt</code></pre>
        </li>
        <li>
          Start learning! Open any notebook in Jupyter:
          <pre><code>jupyter notebook</code></pre>
        </li>
      </ol>

      <h3>Recommended Learning Path</h3>

      <p><strong>For Complete Beginners:</strong></p>
      <ol>
        <li>Start with <code>foundations/programming-and-tools/</code> to get familiar with the essential libraries</li>
        <li>Begin with <code>Linear Regression</code> - it's the foundation of many ML algorithms</li>
        <li>Move to <code>K-Nearest Neighbors</code> - simple yet powerful classification</li>
        <li>Progress to <code>Logistic Regression</code> - introduces probabilistic thinking</li>
        <li>Explore <code>Support Vector Machines</code> - more advanced classification</li>
      </ol>

      <p><strong>For Those with ML Background:</strong></p>
      <ul>
        <li>Jump directly to any algorithm that interests you</li>
        <li>Each tutorial is self-contained with links to prerequisites</li>
      </ul>

      <h2>Key Features</h2>

      <h3>Application-First Approach</h3>
      <p>Every tutorial starts with a real-world problem:</p>
      <ul>
        <li>Predicting stock prices</li>
        <li>Classifying flower species</li>
        <li>Analyzing movie review sentiment</li>
        <li>Detecting face similarity</li>
      </ul>

      <h3>Progressive Complexity</h3>
      <ul>
        <li>Start simple, build up gradually</li>
        <li>Code examples before mathematical formulas</li>
        <li>Theory presented only after seeing it work</li>
      </ul>

      <h3>Comprehensive Explanations</h3>
      <ul>
        <li>Not overly dry or academic</li>
        <li>Plain language with minimal jargon</li>
        <li>Every technical term is explained when introduced</li>
        <li>Visual aids and plots throughout</li>
      </ul>

      <h3>Absolute Correctness</h3>
      <ul>
        <li>All mathematical formulations are precise</li>
        <li>Code examples are tested and verified</li>
        <li>Common misconceptions are explicitly addressed</li>
      </ul>

      <h2>Contributing</h2>

      <p>
        This is a learning project focused on educational clarity and technical accuracy.
        If you find any errors or have suggestions for improvements, please open an issue
        or submit a pull request.
      </p>

      <p>
        Please review <a href="/docs/guidelines">PROJECT_GUIDELINES.md</a> before contributing
        to ensure your additions follow the established teaching philosophy.
      </p>

      <h2>License</h2>

      <p>This project is open source and available for educational purposes.</p>

      <h2>Acknowledgments</h2>

      <ul>
        <li>Built using scikit-learn, NumPy, Pandas, and Matplotlib</li>
        <li>Datasets from UCI Machine Learning Repository and scikit-learn datasets</li>
        <li>Inspired by the belief that machine learning should be accessible to everyone</li>
      </ul>

      <hr />

      <p><strong>Start your ML journey today - no PhD required!</strong></p>
    </>
  );
}
