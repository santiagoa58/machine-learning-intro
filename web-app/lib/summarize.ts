import { cache } from 'react';

/**
 * Server-side content summarization
 * Uses simple excerpt extraction - fast and reliable
 * No AI models loaded server-side to avoid blocking page renders
 */

interface SummaryResult {
  summary: string | null;
  error?: string;
}

/**
 * Generate a summary of content using simple excerpt extraction
 * Cached to avoid repeated processing for the same content
 */
export const generateSummary = cache(async (content: string): Promise<SummaryResult> => {
  // Skip if content is too short
  if (!content || content.length < 100) {
    return { summary: null };
  }

  try {
    const summary = extractExcerpt(content, 200);
    return { summary };
  } catch (error) {
    console.error('Summary generation error:', error);
    return {
      summary: null,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
});

/**
 * Extract a simple excerpt from content as fallback
 * Takes first ~150 characters of meaningful text
 */
export function extractExcerpt(content: string, maxLength: number = 150): string {
  const cleaned = content
    .replace(/^#.*$/gm, '') // Remove headers
    .replace(/```[\s\S]*?```/g, '') // Remove code blocks
    .replace(/`[^`]+`/g, '') // Remove inline code
    .replace(/\n{2,}/g, ' ') // Collapse newlines
    .trim();

  if (cleaned.length <= maxLength) {
    return cleaned;
  }

  // Find a good breaking point (sentence end)
  const truncated = cleaned.slice(0, maxLength);
  const lastPeriod = truncated.lastIndexOf('. ');
  const lastQuestion = truncated.lastIndexOf('? ');
  const lastExclamation = truncated.lastIndexOf('! ');

  const breakPoint = Math.max(lastPeriod, lastQuestion, lastExclamation);

  if (breakPoint > maxLength * 0.6) {
    return truncated.slice(0, breakPoint + 1);
  }

  return truncated + '...';
}
