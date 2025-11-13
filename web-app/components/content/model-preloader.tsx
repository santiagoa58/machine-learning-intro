"use client";
import { Activity } from "react";
import { useEffect } from "react";
import { preloadModel as preloadSummaryModel } from "@/lib/summarize-optimized";

/**
 * Preloads client-side AI model (summary) in the background using Activity
 * Renders as hidden to trigger low-priority model loading
 * Q&A is now server-side, so no need to preload it
 */
export function ModelPreloader() {
  return (
    <Activity mode="hidden">
      <ModelLoader />
    </Activity>
  );
}

function ModelLoader() {
  useEffect(() => {
    // Only preload summary model (Q&A is server-side now)
    preloadSummaryModel();
  }, []);

  // This component doesn't render anything visible
  return null;
}
