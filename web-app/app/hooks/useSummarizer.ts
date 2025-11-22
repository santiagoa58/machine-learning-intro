"use client";
import type { SummarizationPipeline } from "@/lib/types/transformers";
import { filterJoin } from "@/lib/utils";
import { env, pipeline } from "@huggingface/transformers";
import { useCallback, useEffect, useState } from "react";
import {
  createPerformanceMeasure,
  usePerformanceMeasure,
} from "./usePerformanceMeasure";

// Configure ONNX Runtime to suppress verbose warnings
// This prevents the "nodes not assigned to preferred execution providers" warning
env.allowLocalModels = false;
env.useBrowserCache = true;

// Set ONNX Runtime log level to fatal only (suppresses warnings and errors)
// Log levels: "verbose" | "info" | "warning" | "error" | "fatal"
// Setting to "fatal" will suppress the warning about nodes not assigned to EPs
env.backends = {
  onnx: {
    logLevel: "fatal", // Only show fatal errors - suppresses all warnings
    executionProviders: ["wasm"],
  },
};

// Additional browser console filter to catch ONNX Runtime C++ warnings
// This is necessary because ONNX Runtime's C++ logging sometimes bypasses JS log level settings
if (typeof window !== "undefined") {
  const originalConsoleWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    const message = String(args[0] || "");
    // Suppress the specific ONNX Runtime warning about execution provider assignment
    if (
      message.includes("VerifyEachNodeIsAssignedToAnEp") ||
      message.includes("Some nodes were not assigned to the preferred execution providers")
    ) {
      return; // Suppress this specific warning
    }
    originalConsoleWarn.apply(console, args);
  };
}

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
      const model = await pipeline("summarization", undefined, {
        // Configure ONNX Runtime execution providers
        // This suppresses the warning about nodes not being assigned to preferred EPs
        device: "wasm",
        // Set lower log level to suppress verbose warnings
        dtype: "fp32",
      });
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

interface LoadModelOptions {
  subName?: string;
  performance?: ReturnType<typeof createPerformanceMeasure>;
  onError?: (error: unknown) => void;
  onSuccess?: (model: SummarizationPipeline) => void;
}
const loadModel = async ({
  subName,
  performance,
  onError,
  onSuccess,
}: LoadModelOptions = {}) => {
  subName = filterJoin("-", "loadModel", subName);
  performance?.start(subName);
  try {
    const summarizer = await loadSummarizer();
    performance?.end({ subName });
    onSuccess?.(summarizer);
    return summarizer;
  } catch (error) {
    console.error("Failed to load summarization model:", error);
    performance?.end({ subName: "loadModel", error });
    onError?.(error);
    return null;
  }
};

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
  const performance = usePerformanceMeasure("summarization");
  const [isModelLoaded, setIsModelLoaded] = useState(!!summarizerInstance);

  const loadModelWithPerformance = useCallback(
    async (subName?: string) => {
      return await loadModel({
        subName,
        performance,
        onSuccess: () => setIsModelLoaded(true),
        onError: () => setIsModelLoaded(false),
      });
    },
    [performance]
  );

  useEffect(() => {
    // Pre-load the model on mount (deferred to avoid synchronous setState inside effect)
    if (!summarizerInstance && !loadingPromise) {
      // Defer to the next microtask so any setState inside loadModel is not called synchronously in the effect
      void Promise.resolve().then(() =>
        loadModelWithPerformance("initialLoad-useEffect")
      );
    }
  }, [loadModelWithPerformance]);

  const summarize = useCallback(
    async (
      text: string,
      options?: {
        maxLength?: number;
        minLength?: number;
      }
    ): Promise<string | null> => {
      let model: SummarizationPipeline | null = null;
      try {
        model = await loadModelWithPerformance("summarize-hook");
        if (!model) {
          console.warn("Summarization model is not available");
          return null;
        }
      } catch (error) {
        console.error("Failed to load summarization model:", error);
        return null;
      }
      try {
        performance.start("summarize-hook");
        const result = await model(text, {
          max_length: options?.maxLength,
          min_length: options?.minLength,
        });
        performance.end({ subName: "summarize-hook" });
        return result[0]?.summary_text || null;
      } catch (error) {
        console.error("Summarization error:", error);
        performance.end({ subName: "summarize-hook", error });
        return null;
      }
    },
    [performance, loadModelWithPerformance]
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
  const performance = createPerformanceMeasure("summarization");
  try {
    const model = await loadModel({
      subName: "summarizeText-direct",
      performance,
    });
    if (!model) {
      console.warn("Summarization model is not available");
      return null;
    }
    performance.start("summarizeText-direct");
    const result = await model(text, {
      max_length: options?.maxLength,
      min_length: options?.minLength,
    });
    performance.end({ subName: "summarizeText-direct" });
    return result[0]?.summary_text || null;
  } catch (error) {
    console.error("Summarization error:", error);
    performance.end({ subName: "summarizeText-direct", error });
    return null;
  }
}
