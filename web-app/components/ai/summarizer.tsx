"use client";
import { useSummarizer } from "@/app/hooks/useSummarizer";
import { ReactNode, useEffect, useRef, useState } from "react";

export interface SummarizerProps {
  /**
   * Content to wrap and summarize (extracts text automatically)
   */
  children?: ReactNode;

  /**
   * Direct text to summarize (alternative to children)
   */
  text?: string;

  /**
   * Callback when summary is generated
   */
  onSummary?: (summary: string) => void;

  /**
   * Callback when summarization fails
   */
  onError?: (error: Error) => void;

  /**
   * Custom render function for the summary
   */
  renderSummary?: (summary: string, isLoading: boolean) => ReactNode;

  /**
   * Show loading state
   */
  showLoading?: boolean;

  /**
   * Summarization options
   */
  options?: {
    maxLength?: number;
    minLength?: number;
  };

  /**
   * Additional props for the wrapper div
   */
  className?: string;
}

/**
 * Summarizer component - wraps content and displays AI-generated summary
 *
 * @example
 * // Wrap children
 * ```tsx
 * <Summarizer>
 *   <p>Long article text...</p>
 * </Summarizer>
 * ```
 *
 * @example
 * // Direct text
 * ```tsx
 * <Summarizer
 *   text="Long text to summarize"
 *   renderSummary={(summary) => <div className="custom">{summary}</div>}
 * />
 * ```
 *
 * @example
 * // With callback
 * ```tsx
 * <Summarizer
 *   text={content}
 *   onSummary={(summary) => console.log(summary)}
 * />
 * ```
 */
export function Summarizer({
  children,
  text,
  onSummary,
  onError,
  renderSummary,
  showLoading = true,
  options,
  className,
}: SummarizerProps) {
  const [summary, setSummary] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const { summarize, isModelLoaded } = useSummarizer();

  useEffect(() => {
    const runSummarization = async () => {
      // Get text from either prop or children's text content
      const textToSummarize =
        text || contentRef.current?.textContent?.trim() || "";

      if (!textToSummarize || textToSummarize.length < 50) {
        return; // Skip if text is too short
      }

      setIsLoading(true);

      try {
        const result = await summarize(textToSummarize, options);

        if (result) {
          setSummary(result);
          onSummary?.(result);
        }
      } catch (error) {
        console.error("Summarization failed:", error);
        onError?.(error as Error);
      } finally {
        setIsLoading(false);
      }
    };

    // Only run when model is loaded
    if (isModelLoaded) {
      runSummarization();
    }
  }, [text, isModelLoaded, summarize, onSummary, onError, options]);

  // Default summary renderer
  const defaultRenderSummary = (summaryText: string, loading: boolean) => {
    if (loading && showLoading) {
      return (
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800 mb-4">
          <p className="text-sm text-gray-600 dark:text-gray-400 italic">
            ✨ Generating summary...
          </p>
        </div>
      );
    }

    if (!summaryText) return null;

    return (
      <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800 mb-4">
        <p className="text-xs font-semibold text-blue-800 dark:text-blue-300 mb-1">
          📝 TL;DR
        </p>
        <p className="text-sm text-gray-700 dark:text-gray-300">{summaryText}</p>
      </div>
    );
  };

  const summaryElement = renderSummary
    ? renderSummary(summary || "", isLoading)
    : defaultRenderSummary(summary || "", isLoading);

  return (
    <div className={className}>
      {summaryElement}
      {children && <div ref={contentRef}>{children}</div>}
    </div>
  );
}

/**
 * Headless Summarizer - only manages state, no UI
 * Use when you want full control over rendering
 *
 * @example
 * ```tsx
 * <HeadlessSummarizer text="Long text...">
 *   {({ summary, isLoading, error }) => (
 *     <div>
 *       {isLoading && <Spinner />}
 *       {summary && <MyCustomSummary>{summary}</MyCustomSummary>}
 *       {error && <Error>{error.message}</Error>}
 *     </div>
 *   )}
 * </HeadlessSummarizer>
 * ```
 */
export function HeadlessSummarizer({
  text,
  children,
  options,
}: {
  text: string;
  children: (state: {
    summary: string | null;
    isLoading: boolean;
    error: Error | null;
    isModelLoaded: boolean;
  }) => ReactNode;
  options?: { maxLength?: number; minLength?: number };
}) {
  const [summary, setSummary] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const { summarize, isModelLoaded } = useSummarizer();

  useEffect(() => {
    if (!isModelLoaded || !text || text.length < 50) return;

    setIsLoading(true);
    setError(null);

    summarize(text, options)
      .then((result) => {
        setSummary(result);
      })
      .catch((err) => {
        setError(err);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [text, isModelLoaded, summarize, options]);

  return <>{children({ summary, isLoading, error, isModelLoaded })}</>;
}
