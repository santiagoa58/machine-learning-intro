"use client";
import { pipeline } from '@huggingface/transformers';

/**
 * Client-side summarization using transformers.js
 * Integrates with React Suspense for async rendering
 */

interface SummaryCache {
  promise: Promise<string> | null;
  result: string | null;
  error: Error | null;
}

// Global model singleton
let modelPromise: Promise<any> | null = null;
let model: any = null;

// Cache for summaries by content hash
const summaryCache = new Map<string, SummaryCache>();

/**
 * Preload the model in the background (low priority)
 * Can be called from an <Activity mode="hidden"> boundary
 */
export function preloadModel(): void {
  if (!modelPromise && !model) {
    modelPromise = pipeline('summarization').then(m => {
      model = m;
      return m;
    });
  }
}

/**
 * Get the model, loading it if necessary
 * Returns a promise that resolves when the model is ready
 */
async function getModel(): Promise<any> {
  if (model) {
    return model;
  }

  if (!modelPromise) {
    modelPromise = pipeline('summarization').then(m => {
      model = m;
      return m;
    });
  }

  return modelPromise;
}

/**
 * Clean content for summarization
 */
function cleanContent(content: string): string {
  return content
    .replace(/```[\s\S]*?```/g, '') // Remove code blocks
    .replace(/`[^`]+`/g, '') // Remove inline code
    .replace(/\n{3,}/g, '\n\n') // Normalize newlines
    .slice(0, 4000) // Limit length
    .trim();
}

/**
 * Generate a simple hash for content caching
 */
function hashContent(content: string): string {
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return hash.toString(36);
}

/**
 * Summarize content - Suspense-compatible
 * Throws a promise while loading (Suspense pattern)
 * Returns the summary when ready
 */
export function summarizeContent(content: string): string {
  const cleaned = cleanContent(content);
  const cacheKey = hashContent(cleaned);

  // Check cache
  const cached = summaryCache.get(cacheKey);

  // If we have a result, return it
  if (cached?.result) {
    return cached.result;
  }

  // If we have an error, throw it
  if (cached?.error) {
    throw cached.error;
  }

  // If we have a pending promise, throw it (Suspense pattern)
  if (cached?.promise) {
    throw cached.promise;
  }

  // Start new summarization
  const promise = (async () => {
    try {
      const modelInstance = await getModel();
      const result = await modelInstance(cleaned, {
        max_length: 130,
        min_length: 30,
      });

      const summary = result[0]?.summary_text || 'No summary available.';

      // Update cache with result
      summaryCache.set(cacheKey, {
        promise: null,
        result: summary,
        error: null,
      });

      return summary;
    } catch (error) {
      // Update cache with error
      const err = error instanceof Error ? error : new Error('Summarization failed');
      summaryCache.set(cacheKey, {
        promise: null,
        result: null,
        error: err,
      });
      throw err;
    }
  })();

  // Cache the pending promise
  summaryCache.set(cacheKey, {
    promise,
    result: null,
    error: null,
  });

  // Throw the promise (Suspense pattern)
  throw promise;
}

/**
 * Check if model is loaded (for optimistic UI)
 */
export function isModelLoaded(): boolean {
  return model !== null;
}
