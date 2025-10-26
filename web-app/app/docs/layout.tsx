import Link from 'next/link';

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {/* Back to Home */}
        <Link
          href="/"
          className="inline-flex items-center min-h-[44px] py-2 text-blue-600 dark:text-blue-400 hover:underline mb-6 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1"
          aria-label="Navigate back to home page"
        >
          ← Back to Home
        </Link>

        {/* Main Content */}
        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-950/10 dark:border-white/10 p-6 sm:p-8 md:p-12">
          <article className="prose prose-base sm:prose-lg dark:prose-invert max-w-none lg:max-w-4xl mx-auto">
            {children}
          </article>
        </div>

        {/* Documentation Navigation */}
        <nav
          className="mt-6 sm:mt-8 bg-white dark:bg-gray-900 rounded-lg border border-gray-950/10 dark:border-white/10 p-4 sm:p-6"
          aria-label="Documentation navigation"
        >
          <h2
            id="docs-nav-heading"
            className="text-lg font-semibold text-gray-950 dark:text-white mb-4"
          >
            Documentation
          </h2>
          {/* TODO(review): This list duplicates the DOC_PAGES metadata in lib/types.ts and the homepage nav; */}
          {/* TODO(review): render it by mapping over that single source to stay DRY and prevent future drift. */}
          <ul className="space-y-2" aria-labelledby="docs-nav-heading" role="list">
            <li>
              <Link
                href="/docs/readme"
                className="inline-flex items-center min-h-[44px] py-2 text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1"
              >
                Project Overview & Setup
              </Link>
            </li>
            <li>
              <Link
                href="/docs/guidelines"
                className="inline-flex items-center min-h-[44px] py-2 text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1"
              >
                Project Guidelines
              </Link>
            </li>
            <li>
              <Link
                href="/docs/learning-science"
                className="inline-flex items-center min-h-[44px] py-2 text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1"
              >
                Learning Science Review
              </Link>
            </li>
            <li>
              <Link
                href="/docs/improvement-guide"
                className="inline-flex items-center min-h-[44px] py-2 text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1"
              >
                Improvement Guide
              </Link>
            </li>
            <li>
              <Link
                href="/docs/jira"
                className="inline-flex items-center min-h-[44px] py-2 text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1"
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
