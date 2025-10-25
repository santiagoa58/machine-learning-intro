import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <main className="container mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">
            Machine Learning Introduction
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            A comprehensive, hands-on introduction to machine learning that prioritizes
            understanding through application.
          </p>
        </div>

        {/* Philosophy */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Learn by Doing, Understand by Exploring
          </h2>
          <p className="text-gray-600 dark:text-gray-300 mb-4">
            This isn't your typical ML tutorial that starts with pages of mathematics and theory.
            Instead, we believe the best way to learn machine learning is to:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-gray-700 dark:text-gray-300 ml-4">
            <li><strong>See it work first</strong> - Start with a real application and get results</li>
            <li><strong>Understand what it does</strong> - Explore the behavior and capabilities</li>
            <li><strong>Learn why it works</strong> - Dive into the theory with context and motivation</li>
            <li><strong>Master the details</strong> - Deep dive into the mathematics and implementation</li>
          </ol>
        </div>

        {/* Current Status */}
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-400 p-6 mb-12">
          <h3 className="text-lg font-semibold text-yellow-800 dark:text-yellow-300 mb-2">
            🚧 Platform Under Development
          </h3>
          <p className="text-yellow-700 dark:text-yellow-200">
            We're currently building an interactive web platform to transform these tutorials
            into an engaging learning experience with code execution, quizzes, and spaced repetition.
          </p>
        </div>

        {/* Available Content */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {/* Supervised Learning */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              Supervised Learning
            </h3>
            <ul className="space-y-3">
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <div>
                  <strong className="text-gray-900 dark:text-white">Linear Regression</strong>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Predict continuous values</p>
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <div>
                  <strong className="text-gray-900 dark:text-white">K-Nearest Neighbors</strong>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Classify based on proximity</p>
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <div>
                  <strong className="text-gray-900 dark:text-white">Logistic Regression</strong>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Binary classification</p>
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <div>
                  <strong className="text-gray-900 dark:text-white">Support Vector Machines</strong>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Optimal decision boundaries</p>
                </div>
              </li>
            </ul>
          </div>

          {/* Project Documentation */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              Project Documentation
            </h3>
            <ul className="space-y-3">
              <li>
                <Link
                  href="/docs/readme"
                  className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
                >
                  → Project Overview & Setup
                </Link>
              </li>
              <li>
                <Link
                  href="/docs/guidelines"
                  className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
                >
                  → Project Guidelines
                </Link>
              </li>
              <li>
                <Link
                  href="/docs/learning-science"
                  className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
                >
                  → Learning Science Review
                </Link>
              </li>
              <li>
                <Link
                  href="/docs/improvement-guide"
                  className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
                >
                  → Improvement Guide
                </Link>
              </li>
              <li>
                <Link
                  href="/docs/jira"
                  className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
                >
                  → Task Tracker (JIRA)
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Key Features */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Coming Soon: Interactive Features
          </h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                🧠 Active Learning
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Retrieval practice, completion problems, and interactive quizzes
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                💻 Code Execution
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Run Python code directly in your browser with Pyodide
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                📊 Progress Tracking
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Track your learning journey and spaced repetition schedule
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-16 text-gray-600 dark:text-gray-400">
          <p>Built with Next.js 14, Tailwind CSS, and a passion for effective learning</p>
          <p className="mt-2">
            <a
              href="https://github.com/santiagoa58/machine-learning-intro"
              className="text-blue-600 dark:text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              View on GitHub
            </a>
          </p>
        </footer>
      </main>
    </div>
  );
}
