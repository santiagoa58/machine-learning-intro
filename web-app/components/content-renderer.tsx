import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { Content } from '@/lib/content-loader';
import { cn } from '@/lib/utils';

interface ContentRendererProps {
  content: Content;
  className?: string;
}

// Markdown components extracted for performance (avoids recreation on every render)
const markdownComponents: React.ComponentProps<typeof ReactMarkdown>['components'] = {
  h1: ({ node, ...props }) => (
    <h1 className="text-3xl font-bold mb-4" {...props} />
  ),
  h2: ({ node, ...props }) => (
    <h2 className="text-2xl font-semibold mb-3" {...props} />
  ),
  h3: ({ node, ...props }) => (
    <h3 className="text-xl font-semibold mb-2" {...props} />
  ),
  p: ({ node, ...props }) => (
    <p className="mb-4 leading-7" {...props} />
  ),
  strong: ({ node, ...props }) => (
    <strong className="font-semibold" {...props} />
  ),
  em: ({ node, ...props }) => (
    <em className="italic" {...props} />
  ),
  ul: ({ node, ...props }) => (
    <ul className="list-disc pl-6 mb-4 space-y-2" {...props} />
  ),
  ol: ({ node, ...props }) => (
    <ol className="list-decimal pl-6 mb-4 space-y-2" {...props} />
  ),
  li: ({ node, ...props }) => (
    <li className="leading-7" {...props} />
  ),
};

export function ContentRenderer({ content, className }: ContentRendererProps) {
  const { sections } = content.content;

  // Handle empty content
  if (!sections || sections.length === 0) {
    return (
      <article className={cn('prose dark:prose-invert max-w-none', className)}>
        <p className="text-muted-foreground">No content available.</p>
      </article>
    );
  }

  return (
    <article className={cn('prose dark:prose-invert max-w-none space-y-6', className)}>
      {sections.map((section, index) => {
        switch (section.type) {
          case 'text':
            return (
              <div key={index} className="text-section">
                <ReactMarkdown components={markdownComponents}>
                  {section.content}
                </ReactMarkdown>
              </div>
            );

          case 'code':
            return (
              <div key={index} className="code-section">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm font-medium text-muted-foreground uppercase">
                    {section.language}
                  </span>
                </div>
                <SyntaxHighlighter
                  language={section.language}
                  style={oneDark}
                  customStyle={{
                    borderRadius: '0.5rem',
                    padding: '1rem',
                  }}
                  showLineNumbers
                >
                  {section.content}
                </SyntaxHighlighter>
              </div>
            );

          case 'interactive':
            return (
              <div
                key={index}
                className="interactive-section border border-border rounded-lg p-6 bg-card"
              >
                <h3 className="text-lg font-semibold mb-4">
                  Interactive Exercise
                </h3>
                <p className="text-muted-foreground mb-4">
                  Type: {section.exerciseType}
                </p>
                <div className="bg-muted rounded-md p-4">
                  <p className="text-sm font-mono">{section.initialCode}</p>
                </div>
                <p className="text-sm text-muted-foreground mt-4">
                  Full Python editor coming soon with Pyodide integration.
                </p>
              </div>
            );

          default:
            // TypeScript exhaustiveness check
            const _exhaustive: never = section;
            return null;
        }
      })}
    </article>
  );
}
