/**
 * Loading skeleton for content summary
 * Shown while AI generates the summary
 */
export function SummaryLoading() {
  return (
    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800 mb-6 animate-pulse">
      <div className="h-3 bg-blue-200 dark:bg-blue-700 rounded w-20 mb-2"></div>
      <div className="space-y-2">
        <div className="h-3 bg-blue-200 dark:bg-blue-700 rounded w-full"></div>
        <div className="h-3 bg-blue-200 dark:bg-blue-700 rounded w-5/6"></div>
      </div>
    </div>
  );
}
