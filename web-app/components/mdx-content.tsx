'use client';

import { useMDXComponents } from '@/components/mdx-components';
import { MDXRemote, MDXRemoteSerializeResult } from 'next-mdx-remote';
import { serialize } from 'next-mdx-remote/serialize';
import { useEffect, useState } from 'react';
import remarkGfm from 'remark-gfm';

interface MDXContentProps {
  content: string;
}

export function MDXContent({ content }: MDXContentProps) {
  const [mdxSource, setMdxSource] = useState<MDXRemoteSerializeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const components = useMDXComponents({});

  useEffect(() => {
    serialize(content, {
      parseFrontmatter: false,
      mdxOptions: {
        development: process.env.NODE_ENV === 'development',
        remarkPlugins: [remarkGfm],
      },
    })
      .then(setMdxSource)
      .catch((err) => {
        console.error('MDX serialization error:', err);
        setError(err.message);
      });
  }, [content]);

  if (error) {
    return (
      <div className="prose prose-base sm:prose-lg dark:prose-invert max-w-none lg:max-w-3xl mx-auto">
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
          <h3 className="text-red-800 dark:text-red-200 font-semibold mb-2">MDX Rendering Error</h3>
          <p className="text-red-700 dark:text-red-300 text-sm">{error}</p>
          <details className="mt-4">
            <summary className="text-sm cursor-pointer text-red-600 dark:text-red-400">Show raw content</summary>
            <pre className="mt-2 text-xs overflow-x-auto bg-gray-100 dark:bg-gray-800 p-4 rounded">
              {content}
            </pre>
          </details>
        </div>
      </div>
    );
  }

  if (!mdxSource) {
    return (
      <div className="prose prose-base sm:prose-lg dark:prose-invert max-w-none lg:max-w-3xl mx-auto">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="prose prose-base sm:prose-lg dark:prose-invert max-w-none lg:max-w-3xl mx-auto">
      <MDXRemote {...mdxSource} components={components} />
    </div>
  );
}
