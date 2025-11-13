"use client";
import { Component, ReactNode } from "react";

interface SummaryErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface SummaryErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * Error boundary for summary generation
 * Falls back to showing quick overview on error
 */
export class SummaryErrorBoundary extends Component<
  SummaryErrorBoundaryProps,
  SummaryErrorBoundaryState
> {
  constructor(props: SummaryErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): SummaryErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded border border-yellow-200 dark:border-yellow-800 mb-6">
          <p className="text-xs font-semibold text-yellow-800 dark:text-yellow-300 mb-1">
            ⚠️ Summary Unavailable
          </p>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            Could not generate AI summary. The full content is available below.
          </p>
        </div>
      );
    }

    return this.props.children;
  }
}
