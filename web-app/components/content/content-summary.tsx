interface ContentSummaryProps {
  summary: string;
}

/**
 * Server component to display content excerpt
 * Rendered server-side, no client-side JS required
 */
export function ContentSummary({ summary }: ContentSummaryProps) {
  if (!summary) return null;

  return (
    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800 mb-6">
      <p className="text-xs font-semibold text-blue-800 dark:text-blue-300 mb-1">
        📝 Quick Overview
      </p>
      <p className="text-sm text-gray-700 dark:text-gray-300">{summary}</p>
    </div>
  );
}
