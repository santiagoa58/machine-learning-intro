import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { SidebarLayoutContent } from '@/components/layout/sidebar-layout';
import {
  Breadcrumbs,
  BreadcrumbHome,
  BreadcrumbSeparator,
  Breadcrumb,
} from '@/components/layout/breadcrumbs';
import { MDXContent } from '@/components/mdx-content';
import { ClientOnlySummary } from '@/components/content/client-only-summary';
import { ALGORITHMS } from '@/lib/constants';
import { getAlgorithmContent } from '@/lib/content';

export async function generateStaticParams() {
  return ALGORITHMS.map((algo) => ({
    id: algo.id,
  }));
}

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  const algorithm = ALGORITHMS.find((a) => a.id === id);

  if (!algorithm) {
    return {
      title: 'Algorithm Not Found',
    };
  }

  return {
    title: algorithm.name,
    description: `Learn ${algorithm.name} - ${algorithm.description}`,
    openGraph: {
      title: `${algorithm.name} | ML Introduction`,
      description: `Learn ${algorithm.name} - ${algorithm.description}`,
      type: 'article',
    },
  };
}

export default async function AlgorithmPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const algorithm = ALGORITHMS.find((a) => a.id === id);

  if (!algorithm) {
    notFound();
  }

  // Try to load algorithm content
  const algorithmContent = await getAlgorithmContent(id);

  return (
    <SidebarLayoutContent
      breadcrumbs={
        <Breadcrumbs>
          <BreadcrumbHome />
          <BreadcrumbSeparator />
          <Breadcrumb>Supervised Learning</Breadcrumb>
          <BreadcrumbSeparator />
          <Breadcrumb>{algorithm.name}</Breadcrumb>
        </Breadcrumbs>
      }
    >
      <article className="py-6 sm:py-8 lg:py-12">
        {algorithmContent ? (
          <>
            {/* AI-generated summary - client-only to prevent hydration mismatch */}
            <div className="max-w-none lg:max-w-3xl mx-auto mb-6">
              <ClientOnlySummary
                content={algorithmContent.content}
                fallbackSummary={algorithmContent.summary ?? undefined}
              />
            </div>
            {/* Render MDX content */}
            <MDXContent content={algorithmContent.content} />
          </>
        ) : (
          // Fallback to placeholder if content not available
          <div className="prose prose-base sm:prose-lg dark:prose-invert max-w-none lg:max-w-4xl mx-auto">
            <h1>{algorithm.name}</h1>

            <div className="not-prose bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-400 p-4 sm:p-6 mb-8">
              <h2 className="text-lg font-semibold text-blue-800 dark:text-blue-300 mb-2">
                🚧 Tutorial Content Coming Soon
              </h2>
              <p className="text-blue-800 dark:text-blue-100">
                {`This tutorial is currently being developed and will be available soon. Check back later for interactive lessons on ${algorithm.name}.`}
              </p>
            </div>

            <h2>Overview</h2>
            <p>
              <strong>{algorithm.name}</strong> {`is a ${algorithm.category} algorithm used for ${algorithm.description.toLowerCase()}.`}
            </p>

            <h2>{`What You'll Learn`}</h2>
            <ul>
              <li>Understanding the {algorithm.name} algorithm</li>
              <li>Practical applications and use cases</li>
              <li>Implementation with scikit-learn</li>
              <li>Mathematical foundations and theory</li>
              <li>Best practices and common pitfalls</li>
            </ul>

            <h2>Prerequisites</h2>
            <p>Before starting this tutorial, you should be familiar with:</p>
            <ul>
              <li>Basic Python programming</li>
              <li>NumPy and Pandas fundamentals</li>
              <li>Basic statistics and linear algebra</li>
            </ul>

            <div className="not-prose mt-8">
              <a
                href="/docs/readme"
                className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-md text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                ← Back to Documentation
              </a>
            </div>
          </div>
        )}
      </article>
    </SidebarLayoutContent>
  );
}
