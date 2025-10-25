'use client';

import { useEffect } from 'react';
import Link from 'next/link';

export default function DocsError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Documentation page error:', error);
  }, [error]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 px-4 py-8">
      <div className="container mx-auto max-w-2xl">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-8">
          <div className="text-center mb-6">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-red-100 dark:bg-red-900/30 rounded-full mb-4">
              <svg
                className="w-8 h-8 text-red-600 dark:text-red-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Failed to load documentation
            </h1>
            <p className="text-gray-600 dark:text-gray-300">
              {error.message || "We couldn't load this documentation page."}
            </p>
          </div>

          <div className="space-y-3">
            <button
              onClick={reset}
              className="w-full px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              Try loading again
            </button>
            <Link
              href="/docs/readme"
              className="block w-full px-4 py-3 bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-white font-medium rounded-lg text-center transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
            >
              Go to documentation home
            </Link>
            <Link
              href="/"
              className="block w-full text-center px-4 py-3 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              ← Back to main site
            </Link>
          </div>
        </div>

        <nav className="mt-8 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6" aria-label="Other documentation pages">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Browse other documentation
          </h2>
          <ul className="space-y-2">
            <li>
              <Link
                href="/docs/readme"
                className="text-blue-600 dark:text-blue-400 hover:underline"
              >
                Project Overview & Setup
              </Link>
            </li>
            <li>
              <Link
                href="/docs/guidelines"
                className="text-blue-600 dark:text-blue-400 hover:underline"
              >
                Project Guidelines
              </Link>
            </li>
            <li>
              <Link
                href="/docs/learning-science"
                className="text-blue-600 dark:text-blue-400 hover:underline"
              >
                Learning Science Review
              </Link>
            </li>
            <li>
              <Link
                href="/docs/improvement-guide"
                className="text-blue-600 dark:text-blue-400 hover:underline"
              >
                Improvement Guide
              </Link>
            </li>
            <li>
              <Link
                href="/docs/jira"
                className="text-blue-600 dark:text-blue-400 hover:underline"
              >
                Task Tracker (JIRA)
              </Link>
            </li>
          </ul>
        </nav>
      </div>
    </div>
  );
}
