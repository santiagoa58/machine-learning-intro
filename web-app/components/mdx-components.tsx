"use client";

import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Link2 } from "lucide-react";
import type { MDXComponents } from "mdx/types";
import { Highlight, themes } from "prism-react-renderer";

// Import custom MDX components
import { Callout } from "./mdx/callout";
import { Chart } from "./mdx/chart";
import { CodePlayground } from "./mdx/code-playground";
import { DataTable } from "./mdx/data-table";
import { LinearRegressionDemo } from "./mdx/linear-regression-demo";
import { Step, Steps } from "./mdx/steps";
import { Tabs } from "./mdx/tabs";

function createHeading(level: 1 | 2 | 3 | 4 | 5 | 6) {
  const Component = ({
    children,
    ...props
  }: React.HTMLAttributes<HTMLHeadingElement>) => {
    const HeadingTag = `h${level}` as const;
    const id =
      typeof children === "string"
        ? children
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-+|-+$/g, "")
        : undefined;

    // Only add anchor links for h2 and h3
    if (level === 2 || level === 3) {
      return (
        <HeadingTag id={id} className="group/heading scroll-mt-20" {...props}>
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
        </HeadingTag>
      );
    }

    return (
      <HeadingTag id={id} {...props}>
        {children}
      </HeadingTag>
    );
  };

  Component.displayName = `Heading${level}`;
  return Component;
}

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    // Headings with anchor links
    h1: createHeading(1),
    h2: createHeading(2),
    h3: createHeading(3),
    h4: createHeading(4),
    h5: createHeading(5),
    h6: createHeading(6),

    // Code blocks with syntax highlighting
    code: ({
      className,
      children,
      ...props
    }: React.HTMLAttributes<HTMLElement>) => {
      const match = /language-(\w+)/.exec(className || "");
      const isInline = !match;

      if (isInline) {
        return (
          <code
            className="px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-sm font-mono text-pink-600 dark:text-pink-400 not-prose"
            {...props}
          >
            {children}
          </code>
        );
      }

      const language = match[1];
      const code = String(children).replace(/\n$/, "");

      return (
        <Highlight theme={themes.oneDark} code={code} language={language}>
          {({ className, style, tokens, getLineProps, getTokenProps }) => (
            <code className={className} style={style}>
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })}>
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({ token })} />
                  ))}
                </div>
              ))}
            </code>
          )}
        </Highlight>
      );
    },

    // Pre tag (wrapper for code blocks)
    pre: ({ children }: React.HTMLAttributes<HTMLPreElement>) => (
      <pre className="overflow-x-auto rounded-lg my-6">
        {children}
      </pre>
    ),

    // Tables
    table: ({ children }: React.HTMLAttributes<HTMLTableElement>) => (
      <div className="my-6 overflow-x-auto">
        <Table>{children}</Table>
      </div>
    ),
    thead: TableHeader,
    tbody: TableBody,
    tr: TableRow,
    th: ({ children }: React.HTMLAttributes<HTMLTableCellElement>) => (
      <TableHead className="font-semibold">{children}</TableHead>
    ),
    td: ({ children }: React.HTMLAttributes<HTMLTableCellElement>) => (
      <TableCell>{children}</TableCell>
    ),

    // Blockquotes
    blockquote: ({ children }: React.HTMLAttributes<HTMLQuoteElement>) => (
      <blockquote className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 pl-4 py-2 my-4 italic">
        {children}
      </blockquote>
    ),

    // Links
    a: ({
      href,
      children,
      ...props
    }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
      <a
        href={href}
        className="text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 underline"
        {...props}
      >
        {children}
      </a>
    ),

    // Lists
    ul: ({ children }: React.HTMLAttributes<HTMLUListElement>) => (
      <ul className="list-disc list-inside my-4 space-y-2">{children}</ul>
    ),
    ol: ({ children }: React.HTMLAttributes<HTMLOListElement>) => (
      <ol className="list-decimal list-inside my-4 space-y-2">{children}</ol>
    ),
    li: ({ children }: React.HTMLAttributes<HTMLLIElement>) => (
      <li className="text-gray-700 dark:text-gray-300">{children}</li>
    ),

    // Horizontal rule
    hr: () => <hr className="my-8 border-gray-200 dark:border-gray-800" />,

    // Paragraphs
    p: ({ children }: React.HTMLAttributes<HTMLParagraphElement>) => (
      <p className="my-4 text-gray-700 dark:text-gray-300 leading-relaxed">
        {children}
      </p>
    ),

    // Custom components
    Chart,
    CodePlayground,
    Callout,
    DataTable,
    Steps,
    Step,
    Tabs,
    Card,
    LinearRegressionDemo,

    // Allow custom components passed in
    ...components,
  };
}
