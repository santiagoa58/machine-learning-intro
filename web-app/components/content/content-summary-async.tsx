"use client";
import { summarizeContent } from "@/lib/summarize-client";

interface ContentSummaryAsyncProps {
  content: string;
}

/**
 * Client component that uses Suspense to display AI-generated summary
 * Throws a promise while loading (integrates with Suspense boundary)
 */
export function ContentSummaryAsync({ content }: ContentSummaryAsyncProps) {
  // This will throw a promise while loading (Suspense pattern)
  // Or throw an error (Error Boundary pattern)
  // Or return the summary when ready
  const summary = summarizeContent(content);

  return (
    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800 mb-6">
      <p className="text-xs font-semibold text-blue-800 dark:text-blue-300 mb-1">
        ✨ AI Summary
      </p>
      <p className="text-sm text-gray-700 dark:text-gray-300">{summary}</p>
    </div>
  );
}
