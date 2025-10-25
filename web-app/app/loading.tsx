export default function Loading() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="text-center">
        <div className="inline-block relative">
          {/* Spinner */}
          <div
            className="h-16 w-16 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent dark:border-blue-400 dark:border-r-transparent"
            role="status"
            aria-label="Loading"
          >
            <span className="sr-only">Loading...</span>
          </div>
        </div>
        <p className="mt-6 text-lg text-gray-700 dark:text-gray-300 font-medium">
          Loading...
        </p>
        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          Preparing your learning experience
        </p>
      </div>
    </div>
  );
}
