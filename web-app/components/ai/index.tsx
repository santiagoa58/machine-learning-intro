/**
 * AI Components - Production-ready AI features
 *
 * ## Server-Side (Instant)
 * - Content excerpts generated at build time / SSR
 * - Simple text extraction - no AI models, instant rendering
 * - No performance impact on page loads
 *
 * ## Client-Side (User-Triggered)
 * @example Interactive AI Assistant
 * ```tsx
 * import { ContentAssistant } from '@/components/ai';
 *
 * <ContentAssistant content={articleContent} />
 * ```
 *
 * @example Hook usage (custom interactions)
 * ```tsx
 * import { useSummarizer } from '@/components/ai';
 *
 * function MyComponent() {
 *   const { summarize, isModelLoaded } = useSummarizer();
 *
 *   const handleClick = async () => {
 *     const summary = await summarize("Long text here...");
 *     console.log(summary);
 *   };
 *
 *   return <button onClick={handleClick}>Summarize</button>;
 * }
 * ```
 */

// Client-side interactive components
// TODO: Re-enable when content-assistant component is implemented
// export { ContentAssistant } from "./content-assistant";

// Low-level hooks for custom implementations
export { useSummarizer, summarizeText } from "@/app/hooks/useSummarizer";

// Deprecated: Use server-side summaries instead
export { Summarizer, HeadlessSummarizer } from "./summarizer";
export type { SummarizerProps } from "./summarizer";
