import Link from 'next/link';

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Back to Home */}
        <Link
          href="/"
          className="inline-flex items-center text-blue-600 dark:text-blue-400 hover:underline mb-6"
        >
          ← Back to Home
        </Link>

        {/* Main Content */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 md:p-12">
          <article className="prose prose-lg dark:prose-invert max-w-none">
            {children}
          </article>
        </div>

        {/* Documentation Navigation */}
        <nav className="mt-8 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Documentation
          </h3>
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
