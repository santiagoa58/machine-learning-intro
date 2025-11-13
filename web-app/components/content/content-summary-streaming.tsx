"use client";
import { summarizeContentStream } from "@/lib/summarize-optimized";
import { useState, useEffect } from "react";

interface ContentSummaryStreamingProps {
  content: string;
}

/**
 * Client component that displays AI-generated summary with streaming
 */
export function ContentSummaryStreaming({ content }: ContentSummaryStreamingProps) {
  const [summary, setSummary] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function generateSummary() {
      try {
        setIsLoading(true);
        setError(null);
        setSummary("");

        for await (const chunk of summarizeContentStream(content)) {
          if (cancelled) break;
          setSummary(prev => prev + chunk);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to generate summary");
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    generateSummary();

    return () => {
      cancelled = true;
    };
  }, [content]);

  if (error) {
    return null; // Error boundary will handle fallback
  }

  if (!summary && isLoading) {
    return (
      <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800 mb-6 animate-pulse">
        <div className="h-3 bg-blue-200 dark:bg-blue-700 w-20 mb-2 rounded"></div>
        <div className="space-y-2">
          <div className="h-3 bg-blue-200 dark:bg-blue-700 w-full rounded"></div>
          <div className="h-3 bg-blue-200 dark:bg-blue-700 w-5/6 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800 mb-6">
      <p className="text-xs font-semibold text-blue-800 dark:text-blue-300 mb-1 flex items-center gap-2">
        ✨ AI Summary
        {isLoading && (
          <span className="inline-block w-2 h-2 bg-blue-600 rounded-full animate-pulse"></span>
        )}
      </p>
      <p className="text-sm text-gray-700 dark:text-gray-300">
        {summary}
        {isLoading && <span className="inline-block w-1 h-3 ml-1 bg-blue-600 animate-pulse" />}
      </p>
    </div>
  );
}
