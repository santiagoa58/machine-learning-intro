import Link from 'next/link';
import { SidebarLayoutContent } from '@/components/layout/sidebar-layout';
import {
  Breadcrumbs,
  BreadcrumbHome,
  BreadcrumbSeparator,
  Breadcrumb,
} from '@/components/layout/breadcrumbs';
import { ALGORITHMS, DOC_PAGES } from '@/lib/constants';

export default function Home() {
  return (
    <SidebarLayoutContent
      breadcrumbs={
        <Breadcrumbs>
          <BreadcrumbHome />
          <BreadcrumbSeparator />
          <Breadcrumb>Overview</Breadcrumb>
        </Breadcrumbs>
      }
    >
      <div className="mx-auto max-w-7xl py-8 sm:py-12 lg:py-16">
        {/* Header */}
        <header className="text-center mb-12 sm:mb-16">
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-950 dark:text-white mb-4">
            Machine Learning Introduction
          </h1>
          <p className="text-base sm:text-lg lg:text-xl text-gray-700 dark:text-gray-400 max-w-2xl mx-auto">
            A comprehensive, hands-on introduction to machine learning that prioritizes
            understanding through application.
          </p>
        </header>

        {/* Philosophy */}
        <section className="bg-white dark:bg-gray-900 rounded-lg border border-gray-950/10 dark:border-white/10 p-6 sm:p-8 mb-8 sm:mb-12" aria-labelledby="philosophy-heading">
          <h2 id="philosophy-heading" className="text-2xl sm:text-3xl font-bold text-gray-950 dark:text-white mb-4">
            Learn by Doing, Understand by Exploring
          </h2>
          <p className="text-base sm:text-lg text-gray-700 dark:text-gray-400 mb-4">
            {`This isn't your typical ML tutorial that starts with pages of mathematics and theory.
            Instead, we believe the best way to learn machine learning is to:`}
          </p>
          <ol className="list-decimal list-inside space-y-3 text-base sm:text-lg text-gray-700 dark:text-gray-400 ml-2 sm:ml-4">
            <li><strong className="text-gray-950 dark:text-white">See it work first</strong> - Start with a real application and get results</li>
            <li><strong className="text-gray-950 dark:text-white">Understand what it does</strong> - Explore the behavior and capabilities</li>
            <li><strong className="text-gray-950 dark:text-white">Learn why it works</strong> - Dive into the theory with context and motivation</li>
            <li><strong className="text-gray-950 dark:text-white">Master the details</strong> - Deep dive into the mathematics and implementation</li>
          </ol>
        </section>

        {/* Current Status */}
        <aside className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-400 p-4 sm:p-6 mb-8 sm:mb-12" role="note" aria-label="Development status">
          <h3 className="text-base sm:text-lg font-semibold text-yellow-800 dark:text-yellow-300 mb-2">
            🚧 Platform Under Development
          </h3>
          <p className="text-sm sm:text-base text-yellow-800 dark:text-yellow-100">
            {`We're currently building an interactive web platform to transform these tutorials
            into an engaging learning experience with code execution, quizzes, and spaced repetition.`}
          </p>
        </aside>

        {/* Available Content */}
        <div className="grid md:grid-cols-2 gap-6 sm:gap-8 mb-8 sm:mb-12">
          {/* Supervised Learning */}
          <section className="bg-white dark:bg-gray-900 rounded-lg border border-gray-950/10 dark:border-white/10 p-6 sm:p-8" aria-labelledby="supervised-learning-heading">
            <h2 id="supervised-learning-heading" className="text-xl sm:text-2xl font-bold text-gray-950 dark:text-white mb-4">
              Supervised Learning
            </h2>
            <ul className="space-y-3">
              {ALGORITHMS.map((algo) => (
                <li key={algo.id} className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <div>
                    <strong className="text-gray-950 dark:text-white">{algo.name}</strong>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{algo.description}</p>
                  </div>
                </li>
              ))}
            </ul>
          </section>

          {/* Project Documentation */}
          <nav className="bg-white dark:bg-gray-900 rounded-lg border border-gray-950/10 dark:border-white/10 p-6 sm:p-8" aria-labelledby="documentation-heading">
            <h2 id="documentation-heading" className="text-xl sm:text-2xl font-bold text-gray-950 dark:text-white mb-4">
              Project Documentation
            </h2>
            <ul className="space-y-2" role="list">
              {DOC_PAGES.map((page) => (
                <li key={page.slug}>
                  <Link
                    href={page.href}
                    className="inline-flex items-center min-h-[44px] py-2 text-blue-600 dark:text-blue-400 hover:underline font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1"
                  >
                    → {page.title}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </div>

        {/* Key Features */}
        <section className="bg-white dark:bg-gray-900 rounded-lg border border-gray-950/10 dark:border-white/10 p-6 sm:p-8" aria-labelledby="features-heading">
          <h2 id="features-heading" className="text-xl sm:text-2xl font-bold text-gray-950 dark:text-white mb-6">
            Coming Soon: Interactive Features
          </h2>
          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-950 dark:text-white mb-2">
                <span aria-hidden="true">🧠</span> Active Learning
              </h3>
              <p className="text-sm sm:text-base text-gray-700 dark:text-gray-400">
                Retrieval practice, completion problems, and interactive quizzes
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-950 dark:text-white mb-2">
                <span aria-hidden="true">💻</span> Code Execution
              </h3>
              <p className="text-sm sm:text-base text-gray-700 dark:text-gray-400">
                Run Python code directly in your browser with Pyodide
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-950 dark:text-white mb-2">
                <span aria-hidden="true">📊</span> Progress Tracking
              </h3>
              <p className="text-sm sm:text-base text-gray-700 dark:text-gray-400">
                Track your learning journey and spaced repetition schedule
              </p>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center mt-16 text-gray-600 dark:text-gray-400" role="contentinfo">
          <p className="text-sm sm:text-base">
            Built with Next.js 16, React 19, Tailwind CSS v4, and a passion for effective learning
          </p>
          <p className="mt-2">
            <a
              href="https://github.com/santiagoa58/machine-learning-intro"
              className="inline-flex items-center min-h-[44px] text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-2"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="View source code on GitHub (opens in new tab)"
            >
              View on GitHub
            </a>
          </p>
        </footer>
      </div>
    </SidebarLayoutContent>
  );
}
