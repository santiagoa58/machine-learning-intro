/**
 * Type definitions for @huggingface/transformers
 * These types provide better type safety than using 'any'
 */

/**
 * Summarization pipeline result
 */
export interface SummarizationResult {
  summary_text: string;
}

/**
 * Options for summarization pipeline
 */
export interface SummarizationOptions {
  max_length?: number;
  min_length?: number;
  num_beams?: number;
  early_stopping?: boolean;
}

/**
 * Summarization pipeline model
 * Callable interface for the model returned by pipeline('summarization', ...)
 */
export interface SummarizationPipeline {
  (text: string, options?: SummarizationOptions): Promise<SummarizationResult[]>;
}

/**
 * WebGPU Navigator extension
 * Extends Navigator to include the experimental gpu property
 */
declare global {
  interface Navigator {
    gpu?: {
      requestAdapter(): Promise<unknown>;
    };
  }
}

export {};
