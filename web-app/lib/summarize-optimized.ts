"use client";
import { pipeline, env } from '@huggingface/transformers';
import type { SummarizationPipeline } from './types/transformers';
import './types/transformers'; // Import for Navigator.gpu type extension

/**
 * Optimized client-side summarization with streaming
 * Uses T5-small for reliable, fast summaries
 */

// Enable WebGPU if available
if (typeof window !== 'undefined' && navigator.gpu) {
  env.backends.onnx.wasm!.proxy = false;
}

// Singleton model
let modelPromise: Promise<SummarizationPipeline> | null = null;
let model: SummarizationPipeline | null = null;

// Summary cache
const summaryCache = new Map<string, string>();

/**
 * Hash function for content
 */
function hashContent(content: string): string {
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash.toString(36);
}

/**
 * Get the summarization model
 */
async function getModel(): Promise<SummarizationPipeline> {
  if (model) return model;

  if (!modelPromise) {
    modelPromise = pipeline(
      'summarization',
      'Xenova/t5-small',
      {
        device: navigator.gpu ? 'webgpu' : 'wasm',
        dtype: 'fp32', // Use fp32 for better quality
      }
    ).then(m => {
      model = m as unknown as SummarizationPipeline;
      return model;
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
    .replace(/[*_]{1,2}/g, '') // Remove emphasis markers
    .slice(0, 1500) // Limit length
    .trim();
}

/**
 * Generate summary with streaming
 */
export async function* summarizeContentStream(
  content: string
): AsyncGenerator<string, void, unknown> {
  const cacheKey = hashContent(content);

  // Return cached result
  if (summaryCache.has(cacheKey)) {
    const cached = summaryCache.get(cacheKey)!;
    // Simulate streaming for smooth UX
    const words = cached.split(' ');
    for (const word of words) {
      yield word + ' ';
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    return;
  }

  try {
    const cleaned = prepareContent(content);

    if (!cleaned || cleaned.length < 50) {
      const msg = 'Content is too short to summarize.';
      summaryCache.set(cacheKey, msg);
      yield msg;
      return;
    }

    const modelInstance = await getModel();

    // Generate summary
    const result = await modelInstance(cleaned, {
      max_length: 100,
      min_length: 30,
      num_beams: 2,
      early_stopping: true,
    });

    let summary = result[0]?.summary_text || 'Unable to generate summary.';

    // Clean up summary
    summary = summary
      .replace(/^summarize:\s*/i, '')
      .trim();

    // Cache the result
    summaryCache.set(cacheKey, summary);

    // Stream the summary word by word
    const words = summary.split(' ');
    for (const word of words) {
      yield word + ' ';
      await new Promise(resolve => setTimeout(resolve, 20));
    }
  } catch (error) {
    const err = error instanceof Error ? error : new Error('Failed to generate summary');
    console.error('Summary error:', err);
    throw err;
  }
}

/**
 * Generate summary without streaming
 */
export async function summarizeContent(content: string): Promise<string> {
  let fullSummary = '';
  for await (const chunk of summarizeContentStream(content)) {
    fullSummary += chunk;
  }
  return fullSummary.trim();
}

/**
 * Preload the model
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

/**
 * Clear summary cache
 */
export function clearSummaryCache(): void {
  summaryCache.clear();
}
