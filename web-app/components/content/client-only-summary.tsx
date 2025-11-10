"use client";
import { useState, useEffect } from "react";
import { ContentSummaryStreaming } from "./content-summary-streaming";
import { SummaryErrorBoundary } from "./summary-error";
import { ContentSummary } from "./content-summary";

interface ClientOnlySummaryProps {
  content: string;
  fallbackSummary?: string;
}

/**
 * Client-only wrapper with streaming support
 * Prevents hydration mismatches and provides smooth streaming UX
 */
export function ClientOnlySummary({ content, fallbackSummary }: ClientOnlySummaryProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // During SSR and initial hydration, don't render anything
  if (!mounted) {
    return null;
  }

  // After mount, render the streaming AI summary
  return (
    <SummaryErrorBoundary
      fallback={
        fallbackSummary ? (
          <ContentSummary summary={fallbackSummary} />
        ) : null
      }
    >
      <ContentSummaryStreaming content={content} />
    </SummaryErrorBoundary>
  );
}
