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

// Markdown components will use .prose styles from typography.css
// No need for custom components - Compass prose handles everything

export function ContentRenderer({ content, className }: ContentRendererProps) {
  const { sections } = content.content;

  // Handle empty content
  if (!sections || sections.length === 0) {
    return (
      <article className={cn('prose max-w-none', className)}>
        <p className="text-gray-400 dark:text-gray-600">No content available.</p>
      </article>
    );
  }

  return (
    <article className={cn('prose max-w-none space-y-6', className)}>
      {sections.map((section, index) => {
        switch (section.type) {
          case 'text':
            return (
              <div key={index} className="text-section">
                <ReactMarkdown>{section.content}</ReactMarkdown>
              </div>
            );

          case 'code':
            return (
              <div key={index} className="code-section">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600 uppercase dark:text-gray-400">
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
                className="interactive-section border border-gray-950/10 rounded-lg p-6 bg-white dark:border-white/10 dark:bg-gray-900"
              >
                <h3 className="text-lg font-semibold mb-4 text-gray-950 dark:text-white">
                  Interactive Exercise
                </h3>
                <p className="text-gray-600 mb-4 dark:text-gray-400">
                  Type: {section.exerciseType}
                </p>
                <div className="bg-gray-100 rounded-md p-4 dark:bg-gray-800">
                  <p className="text-sm font-mono text-gray-950 dark:text-white">{section.initialCode}</p>
                </div>
                <p className="text-sm text-gray-600 mt-4 dark:text-gray-400">
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
