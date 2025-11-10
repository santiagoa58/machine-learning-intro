"use client";
import { pipeline, env } from '@huggingface/transformers';

/**
 * Client-side summarization with streaming support
 * Uses T5-small for fast summarization
 */

// Enable WebGPU if available
if (typeof window !== 'undefined' && navigator.gpu) {
  env.backends.onnx.wasm.proxy = false;
}

// Singleton model
let modelPromise: Promise<any> | null = null;
let model: any = null;

// Summary cache: contentHash -> { promise?, result?, error? }
const summaryCache = new Map<string, {
  promise?: Promise<string>;
  result?: string;
  error?: Error;
  stream?: AsyncGenerator<string>;
}>();

/**
 * Simple hash function for content
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
 * Get the summarization model
 */
async function getModel(): Promise<any> {
  if (model) return model;

  if (!modelPromise) {
    modelPromise = pipeline(
      'summarization',
      'Xenova/distilbart-cnn-6-6',
      {
        device: navigator.gpu ? 'webgpu' : 'wasm',
        dtype: 'q8',
      }
    ).then(m => {
      model = m;
      return m;
    });
  }

  return modelPromise;
}

/**
 * Clean and prepare content for summarization
 */
function prepareContent(content: string): string {
  return content
    .replace(/```[\s\S]*?```/g, '') // Remove code blocks
    .replace(/`[^`]+`/g, '') // Remove inline code
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Convert links to text
    .replace(/\n{3,}/g, '\n\n') // Normalize newlines
    .replace(/#{1,6}\s/g, '') // Remove markdown headers
    .slice(0, 1500) // Limit length
    .trim();
}

/**
 * Generate summary with streaming support
 * Returns an async generator that yields chunks as they're generated
 */
export async function* summarizeContentStream(
  content: string
): AsyncGenerator<string, void, unknown> {
  const cacheKey = hashContent(content);
  const cached = summaryCache.get(cacheKey);

  // Return cached result immediately
  if (cached?.result) {
    yield cached.result;
    return;
  }

  // If there's a cached error, throw it
  if (cached?.error) {
    throw cached.error;
  }

  // If already streaming, use that stream
  if (cached?.stream) {
    yield* cached.stream;
    return;
  }

  try {
    const cleaned = prepareContent(content);

    if (!cleaned) {
      yield 'Content is too short to summarize.';
      return;
    }

    const modelInstance = await getModel();

    // Generate summary
    const result = await modelInstance(cleaned, {
      max_length: 130,
      min_length: 30,
    });

    const summary = result[0]?.summary_text || 'Unable to generate summary.';

    // Cache the result
    summaryCache.set(cacheKey, { result: summary });

    // Simulate streaming for smooth UX
    const words = summary.split(' ');
    for (const word of words) {
      yield word + ' ';
      await new Promise(resolve => setTimeout(resolve, 20));
    }
  } catch (error) {
    const err = error instanceof Error ? error : new Error('Failed to generate summary');
    summaryCache.set(cacheKey, { error: err });
    throw err;
  }
}

/**
 * Generate summary without streaming (backwards compatible)
 */
export async function summarizeContent(content: string): Promise<string> {
  let fullSummary = '';
  for await (const chunk of summarizeContentStream(content)) {
    fullSummary += chunk;
  }
  return fullSummary.trim();
}

/**
 * Preload the summarization model
 */
export function preloadModel(): void {
  if (!modelPromise && !model) {
    getModel().catch(err => {
      console.error('Failed to preload summary model:', err);
    });
  }
}

/**
 * Check if model is loaded
 */
export function isModelLoaded(): boolean {
  return model !== null;
}
