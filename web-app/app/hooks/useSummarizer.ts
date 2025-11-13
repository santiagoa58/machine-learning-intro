"use client";
import { pipeline } from "@huggingface/transformers";
import { useCallback, useEffect, useState } from "react";
import type { SummarizationPipeline } from "@/lib/types/transformers";

// Singleton to ensure model loads only once across entire app
let summarizerInstance: SummarizationPipeline | null = null;
let loadingPromise: Promise<SummarizationPipeline> | null = null;

/**
 * Loads the summarization model (singleton)
 * Uses distilbart-cnn-6-6 by default (smaller, faster)
 */
async function loadSummarizer(): Promise<SummarizationPipeline> {
  if (summarizerInstance) {
    return summarizerInstance;
  }

  if (loadingPromise) {
    return loadingPromise;
  }

  loadingPromise = (async () => {
    try {
      const model = await pipeline("summarization");
      summarizerInstance = model as unknown as SummarizationPipeline;
      return summarizerInstance;
    } catch (error) {
      console.error("❌ Failed to load summarization model:", error);
      loadingPromise = null; // Reset so we can retry
      throw error;
    }
  })();

  return loadingPromise;
}

/**
 * Hook that provides a function to summarize text
 *
 * @example
 * ```tsx
 * const summarize = useSummarizer();
 *
 * const handleClick = async () => {
 *   const result = await summarize("Long text here...");
 *   console.log(result);
 * };
 * ```
 */
export function useSummarizer() {
  const [isModelLoaded, setIsModelLoaded] = useState(!!summarizerInstance);

  useEffect(() => {
    // Pre-load the model on mount
    if (!summarizerInstance && !loadingPromise) {
      loadSummarizer()
        .then(() => setIsModelLoaded(true))
        .catch(() => setIsModelLoaded(false));
    }
  }, []);

  const summarize = useCallback(
    async (
      text: string,
      options?: {
        maxLength?: number;
        minLength?: number;
      }
    ): Promise<string | null> => {
      try {
        const model = await loadSummarizer();
        const result = await model(text, {
          max_length: options?.maxLength,
          min_length: options?.minLength,
        });
        return result[0]?.summary_text || null;
      } catch (error) {
        console.error("Summarization error:", error);
        return null;
      }
    },
    []
  );

  return { summarize, isModelLoaded };
}

/**
 * Direct function to summarize text (no hook)
 * Useful for one-off summarizations
 */
export async function summarizeText(
  text: string,
  options?: { maxLength?: number; minLength?: number }
): Promise<string | null> {
  try {
    const model = await loadSummarizer();
    const result = await model(text, {
      max_length: options?.maxLength,
      min_length: options?.minLength,
    });
    return result[0]?.summary_text || null;
  } catch (error) {
    console.error("Summarization error:", error);
    return null;
  }
}
