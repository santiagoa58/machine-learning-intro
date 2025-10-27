import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import { Link2 } from 'lucide-react';

interface MarkdownContentProps {
  content: string;
}

export function MarkdownContent({ content }: MarkdownContentProps) {
  const components: Components = {
    // Custom code block rendering
    code({ inline, className, children, ...props }) {
      return !inline ? (
        <pre className="bg-gray-900 dark:bg-gray-950 text-gray-100 rounded-md p-4 overflow-x-auto my-4">
          <code className={className} {...props}>
            {children}
          </code>
        </pre>
      ) : (
        <code
          className="px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-sm font-mono"
          {...props}
        >
          {children}
        </code>
      );
    },

    // Custom table rendering
    table({ children }) {
      return (
        <div className="overflow-x-auto my-6">
          <table className="min-w-full divide-y divide-gray-300 dark:divide-gray-700">
            {children}
          </table>
        </div>
      );
    },

    // Custom heading anchors for better navigation
    h2({ children }) {
      const id = String(children)
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, ''); // Remove leading/trailing dashes
      return (
        <h2 id={id} className="group/heading scroll-mt-20">
          <a
            href={`#${id}`}
            className="inline-flex items-center gap-2 no-underline hover:no-underline focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 rounded px-1 -ml-1"
          >
            <span>{children}</span>
            <Link2
              className="w-4 h-4 shrink-0 text-gray-400 opacity-0 group-hover/heading:opacity-100 transition-opacity"
              aria-hidden="true"
            />
          </a>
        </h2>
      );
    },

    h3({ children }) {
      const id = String(children)
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, ''); // Remove leading/trailing dashes
      return (
        <h3 id={id} className="group/heading scroll-mt-20">
          <a
            href={`#${id}`}
            className="inline-flex items-center gap-2 no-underline hover:no-underline focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 rounded px-1 -ml-1"
          >
            <span>{children}</span>
            <Link2
              className="w-4 h-4 shrink-0 text-gray-400 opacity-0 group-hover/heading:opacity-100 transition-opacity"
              aria-hidden="true"
            />
          </a>
        </h3>
      );
    },

    // Custom blockquote styling
    blockquote({ children }) {
      return (
        <blockquote className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 pl-4 py-2 my-4 italic">
          {children}
        </blockquote>
      );
    },
  };

  return (
    <div className="prose prose-base sm:prose-lg prose-gray dark:prose-invert max-w-none lg:max-w-4xl mx-auto">
      <ReactMarkdown components={components}>{content}</ReactMarkdown>
    </div>
  );
}
