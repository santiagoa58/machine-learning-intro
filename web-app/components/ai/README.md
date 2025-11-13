# AI Components

Production-ready AI features with server-side generation and client-side interactivity.

## Architecture

### Server-Side (Instant)
- **Content excerpts** generated at build time / SSR
- Simple text extraction - fast and reliable
- No AI models loaded server-side (instant page loads)
- React `cache()` for efficient caching

### Client-Side (User-Triggered)
- **Interactive AI assistant** for Q&A and explanations
- Browser-based ML using transformers.js
- Only loads when user clicks the assistant button
- No automatic model loading on page load

## Features

- ✅ **Fast page loads** - Server excerpts are instant, no blocking AI
- ✅ **Efficient caching** - Server excerpts cached, client models singleton
- ✅ **User-driven** - AI only loads when user wants it
- ✅ **TypeScript** - Fully typed
- ✅ **Zero config** - Works out of the box

## Setup

No setup required! Everything works out of the box.

### Dependencies

Already installed:
- `@huggingface/transformers` - Works both server-side and client-side
- React 19+ with Server Components

## CSP Configuration

Add to your `next.config.ts`:

```ts
{
  headers: [{
    source: '/:path*',
    headers: [{
      key: 'Content-Security-Policy',
      value: [
        // ... other directives
        `script-src 'self' https://cdn.jsdelivr.net 'unsafe-eval' 'unsafe-inline'`,
        "connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co https://cdn.jsdelivr.net",
      ].join('; ')
    }]
  }]
}
```

## Usage

### 1. Hook API (Most Flexible)

```tsx
import { useSummarizer } from '@/components/ai';

function MyComponent() {
  const { summarize, isModelLoaded } = useSummarizer();
  const [summary, setSummary] = useState<string | null>(null);

  const handleClick = async () => {
    const result = await summarize("Long text here...", {
      maxLength: 130,
      minLength: 30
    });
    setSummary(result);
  };

  return (
    <div>
      <button onClick={handleClick} disabled={!isModelLoaded}>
        Summarize
      </button>
      {summary && <p>{summary}</p>}
    </div>
  );
}
```

### 2. Component API (Easiest)

```tsx
import { Summarizer } from '@/components/ai';

// Wrap existing content
<Summarizer>
  <article>
    <p>Long article text...</p>
  </article>
</Summarizer>

// Direct text
<Summarizer text="Long text to summarize..." />

// Custom rendering
<Summarizer
  text={content}
  renderSummary={(summary, isLoading) => (
    <div className="custom-summary">
      {isLoading ? <Spinner /> : <p>{summary}</p>}
    </div>
  )}
/>

// With callbacks
<Summarizer
  text={content}
  onSummary={(summary) => console.log('Summary:', summary)}
  onError={(error) => console.error('Error:', error)}
/>
```

### 3. Headless API (Full Control)

```tsx
import { HeadlessSummarizer } from '@/components/ai';

<HeadlessSummarizer text="Long text...">
  {({ summary, isLoading, error, isModelLoaded }) => (
    <div>
      {!isModelLoaded && <p>Loading AI model...</p>}
      {isLoading && <Spinner />}
      {error && <Error>{error.message}</Error>}
      {summary && <CustomCard>{summary}</CustomCard>}
    </div>
  )}
</HeadlessSummarizer>
```

### 4. Direct Function (No React)

```tsx
import { summarizeText } from '@/components/ai';

const summary = await summarizeText("Long text...", {
  maxLength: 100,
  minLength: 20
});
```

## Props

### `useSummarizer()`

Returns:
- `summarize: (text: string, options?) => Promise<string | null>`
- `isModelLoaded: boolean`

### `<Summarizer>`

| Prop | Type | Description |
|------|------|-------------|
| `children` | `ReactNode` | Content to wrap and summarize |
| `text` | `string` | Direct text to summarize |
| `onSummary` | `(summary: string) => void` | Callback when summary is generated |
| `onError` | `(error: Error) => void` | Callback when summarization fails |
| `renderSummary` | `(summary: string, isLoading: boolean) => ReactNode` | Custom render function |
| `showLoading` | `boolean` | Show loading state (default: true) |
| `options` | `{ maxLength?: number, minLength?: number }` | Summarization options |
| `className` | `string` | Additional CSS classes |

## Performance

- **First load**: ~1-2 minutes (downloads ~80MB model)
- **Subsequent loads**: Instant (cached in browser)
- **Summarization**: ~2-5 seconds (depends on text length)

## Model

Uses `Xenova/distilbart-cnn-6-6`:
- Optimized for browser (ONNX format)
- Smaller than full BART (~80MB vs 500MB)
- Good quality summaries
- Runs on CPU (WASM)

## Troubleshooting

### CSP Errors
Add domains to your CSP configuration (see above)

### Model not loading
Check browser console for:
1. CSP violations
2. Network errors
3. Browser compatibility (needs modern browser with WASM support)

### Slow performance
- First load is always slow (downloading model)
- Consider showing a prominent loading indicator
- Model is cached after first load

## Examples

See `components/ai/transformer.tsx` for a complete demo with multiple usage patterns.
