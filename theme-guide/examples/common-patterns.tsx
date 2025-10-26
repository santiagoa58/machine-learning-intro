/**
 * Common UI Patterns - Code Examples
 *
 * This file contains real-world examples of common UI patterns
 * following the Compass design system.
 */

// =============================================================================
// BUTTONS
// =============================================================================

// Primary Button (Default)
export function PrimaryButton() {
  return (
    <button className="inline-flex items-center justify-center gap-2 rounded-full bg-gray-950 px-3.5 py-2 text-sm/6 font-semibold text-white transition-colors hover:bg-gray-800 focus:outline-2 focus:outline-offset-2 focus:outline-blue-500 disabled:pointer-events-none disabled:opacity-50 dark:bg-gray-700 dark:hover:bg-gray-600">
      Primary Action
    </button>
  );
}

// Destructive Button
export function DestructiveButton() {
  return (
    <button className="inline-flex items-center justify-center gap-2 rounded-full bg-red-600 px-3.5 py-2 text-sm/6 font-semibold text-white transition-colors hover:bg-red-700 focus:outline-2 focus:outline-offset-2 focus:outline-blue-500 dark:bg-red-700 dark:hover:bg-red-800">
      Delete
    </button>
  );
}

// Secondary Button
export function SecondaryButton() {
  return (
    <button className="inline-flex items-center justify-center gap-2 rounded-full bg-gray-100 px-3.5 py-2 text-sm/6 font-semibold text-gray-950 transition-colors hover:bg-gray-200 focus:outline-2 focus:outline-offset-2 focus:outline-blue-500 dark:bg-gray-800 dark:text-white dark:hover:bg-gray-700">
      Secondary
    </button>
  );
}

// Outline Button
export function OutlineButton() {
  return (
    <button className="inline-flex items-center justify-center gap-2 rounded-full border border-gray-950/10 bg-white px-3.5 py-2 text-sm/6 font-semibold transition-colors hover:bg-gray-50 focus:outline-2 focus:outline-offset-2 focus:outline-blue-500 dark:border-white/10 dark:bg-gray-950 dark:hover:bg-gray-900">
      Outline
    </button>
  );
}

// Ghost Button
export function GhostButton() {
  return (
    <button className="inline-flex items-center justify-center gap-2 rounded-full px-3.5 py-2 text-sm/6 font-semibold transition-colors hover:bg-gray-100 focus:outline-2 focus:outline-offset-2 focus:outline-blue-500 dark:hover:bg-gray-800">
      Ghost
    </button>
  );
}

// Button Sizes
export function ButtonSizes() {
  return (
    <div className="flex items-center gap-2">
      {/* Small */}
      <button className="rounded-full bg-gray-950 px-3 py-1.5 text-xs font-semibold text-white">
        Small
      </button>

      {/* Default */}
      <button className="rounded-full bg-gray-950 px-3.5 py-2 text-sm/6 font-semibold text-white">
        Default
      </button>

      {/* Large */}
      <button className="rounded-full bg-gray-950 px-4 py-2.5 text-sm/6 font-semibold text-white">
        Large
      </button>
    </div>
  );
}

// =============================================================================
// CARDS
// =============================================================================

// Basic Card
export function BasicCard() {
  return (
    <div className="rounded-lg border border-gray-950/10 bg-white p-6 dark:border-white/10 dark:bg-gray-900">
      <h3 className="text-lg font-semibold text-gray-950 dark:text-white">
        Card Title
      </h3>
      <p className="mt-2 text-sm text-gray-700 dark:text-gray-400">
        Card content goes here. This follows the Compass design system.
      </p>
    </div>
  );
}

// Card with Header
export function CardWithHeader() {
  return (
    <div className="rounded-lg border border-gray-950/10 bg-white dark:border-white/10 dark:bg-gray-900">
      <div className="border-b border-gray-950/10 px-6 py-4 dark:border-white/10">
        <h3 className="text-lg font-semibold text-gray-950 dark:text-white">
          Card Header
        </h3>
      </div>
      <div className="p-6">
        <p className="text-sm text-gray-700 dark:text-gray-400">
          Card content with separate header section.
        </p>
      </div>
    </div>
  );
}

// Clickable Card
export function ClickableCard() {
  return (
    <a
      href="#"
      className="block rounded-lg border border-gray-950/10 bg-white p-6 transition-all hover:border-gray-950/20 hover:shadow-lg dark:border-white/10 dark:bg-gray-900 dark:hover:border-white/20"
    >
      <h3 className="text-lg font-semibold text-gray-950 dark:text-white">
        Clickable Card
      </h3>
      <p className="mt-2 text-sm text-gray-700 dark:text-gray-400">
        Hover to see the effect. Border and shadow change on hover.
      </p>
    </a>
  );
}

// =============================================================================
// FORMS
// =============================================================================

// Text Input
export function TextInput() {
  return (
    <input
      type="text"
      className="w-full rounded-md border border-gray-950/10 bg-white px-3 py-2 text-sm placeholder:text-gray-400 focus:outline-2 focus:outline-offset-2 focus:outline-blue-500 dark:border-white/10 dark:bg-gray-900 dark:text-white"
      placeholder="Enter text..."
    />
  );
}

// Textarea
export function Textarea() {
  return (
    <textarea
      className="w-full rounded-md border border-gray-950/10 bg-white px-3 py-2 text-sm placeholder:text-gray-400 focus:outline-2 focus:outline-offset-2 focus:outline-blue-500 dark:border-white/10 dark:bg-gray-900 dark:text-white"
      rows={4}
      placeholder="Enter text..."
    />
  );
}

// Label + Input
export function LabeledInput() {
  return (
    <div className="space-y-2">
      <label
        htmlFor="email"
        className="block text-sm font-medium text-gray-950 dark:text-white"
      >
        Email Address
      </label>
      <input
        id="email"
        type="email"
        className="w-full rounded-md border border-gray-950/10 bg-white px-3 py-2 text-sm placeholder:text-gray-400 focus:outline-2 focus:outline-offset-2 focus:outline-blue-500 dark:border-white/10 dark:bg-gray-900 dark:text-white"
        placeholder="you@example.com"
      />
    </div>
  );
}

// Form with Error State
export function InputWithError() {
  return (
    <div className="space-y-2">
      <label
        htmlFor="username"
        className="block text-sm font-medium text-gray-950 dark:text-white"
      >
        Username
      </label>
      <input
        id="username"
        type="text"
        className="w-full rounded-md border border-red-600 bg-white px-3 py-2 text-sm placeholder:text-gray-400 focus:outline-2 focus:outline-offset-2 focus:outline-red-600 dark:bg-gray-900 dark:text-white"
        placeholder="Enter username..."
      />
      <p className="text-sm text-red-600 dark:text-red-400">
        Username is already taken.
      </p>
    </div>
  );
}

// =============================================================================
// NAVIGATION
// =============================================================================

// Sidebar Navigation
export function SidebarNav() {
  return (
    <nav aria-label="Course navigation">
      <div className="space-y-8">
        <div>
          <h2 className="text-sm font-semibold text-gray-950 dark:text-white">
            Module 1: Introduction
          </h2>
          <ul className="mt-3 flex flex-col gap-3 border-l border-gray-950/10 text-sm text-gray-700 dark:border-white/10 dark:text-gray-400">
            <li className="-ml-px flex border-l border-transparent pl-4 hover:border-gray-400 hover:text-gray-950 aria-[current=page]:border-gray-950 aria-[current=page]:font-medium aria-[current=page]:text-gray-950 dark:hover:text-white dark:aria-[current=page]:border-white dark:aria-[current=page]:text-white">
              <a href="#" aria-current="page">
                Getting Started
              </a>
            </li>
            <li className="-ml-px flex border-l border-transparent pl-4 hover:border-gray-400 hover:text-gray-950 dark:hover:text-white">
              <a href="#">What is Machine Learning?</a>
            </li>
            <li className="-ml-px flex border-l border-transparent pl-4 hover:border-gray-400 hover:text-gray-950 dark:hover:text-white">
              <a href="#">Setup Your Environment</a>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}

// Breadcrumbs
export function Breadcrumbs() {
  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-2 text-sm">
      <a
        href="#"
        className="text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-white"
      >
        Home
      </a>
      <span className="text-gray-400">/</span>
      <a
        href="#"
        className="text-gray-700 hover:text-gray-950 dark:text-gray-400 dark:hover:text-white"
      >
        Module 1
      </a>
      <span className="text-gray-400">/</span>
      <span className="text-gray-950 dark:text-white">Getting Started</span>
    </nav>
  );
}

// =============================================================================
// CONTENT / PROSE
// =============================================================================

// Prose Content (Markdown/Rich Text)
export function ProseContent() {
  return (
    <article className="prose max-w-none">
      <h1>Introduction to Machine Learning</h1>
      <p>
        Machine learning is a subset of artificial intelligence that focuses on
        building systems that learn from data.
      </p>
      <h2>What You'll Learn</h2>
      <ul>
        <li>Supervised learning algorithms</li>
        <li>Unsupervised learning techniques</li>
        <li>Model evaluation and validation</li>
      </ul>
      <h3>Prerequisites</h3>
      <p>
        Before starting, you should have basic knowledge of{" "}
        <strong>Python programming</strong> and{" "}
        <strong>linear algebra</strong>.
      </p>
      <pre>
        <code>
          {`# Example: Simple linear regression
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)
print(f"Coefficient: {model.coef_[0]}")`}
        </code>
      </pre>
    </article>
  );
}

// Code Block (without prose)
export function CodeBlock() {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium uppercase text-gray-600 dark:text-gray-400">
          Python
        </span>
      </div>
      <pre className="overflow-x-auto rounded-lg bg-gray-950 p-4">
        <code className="text-sm text-white">
          {`def hello_world():
    print("Hello, World!")

hello_world()`}
        </code>
      </pre>
    </div>
  );
}

// =============================================================================
// ALERTS / NOTIFICATIONS
// =============================================================================

// Info Alert
export function InfoAlert() {
  return (
    <div className="rounded-md border border-blue-500/20 bg-blue-50 p-4 dark:bg-blue-950/20">
      <div className="flex items-start gap-3">
        <div className="flex-1">
          <h4 className="text-sm font-semibold text-blue-950 dark:text-blue-300">
            Information
          </h4>
          <p className="mt-1 text-sm text-blue-900 dark:text-blue-400">
            This is an informational message.
          </p>
        </div>
      </div>
    </div>
  );
}

// Success Alert
export function SuccessAlert() {
  return (
    <div className="rounded-md border border-green-500/20 bg-green-50 p-4 dark:bg-green-950/20">
      <div className="flex items-start gap-3">
        <div className="flex-1">
          <h4 className="text-sm font-semibold text-green-950 dark:text-green-300">
            Success
          </h4>
          <p className="mt-1 text-sm text-green-900 dark:text-green-400">
            Your changes have been saved successfully.
          </p>
        </div>
      </div>
    </div>
  );
}

// Error Alert
export function ErrorAlert() {
  return (
    <div className="rounded-md border border-red-500/20 bg-red-50 p-4 dark:bg-red-950/20">
      <div className="flex items-start gap-3">
        <div className="flex-1">
          <h4 className="text-sm font-semibold text-red-950 dark:text-red-300">
            Error
          </h4>
          <p className="mt-1 text-sm text-red-900 dark:text-red-400">
            Something went wrong. Please try again.
          </p>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// LAYOUTS
// =============================================================================

// Page Header
export function PageHeader() {
  return (
    <div className="border-b border-gray-950/10 pb-5 dark:border-white/10">
      <h1 className="text-3xl font-bold text-gray-950 dark:text-white">
        Getting Started with ML
      </h1>
      <p className="mt-2 text-sm text-gray-700 dark:text-gray-400">
        Learn the fundamentals of machine learning in this comprehensive guide.
      </p>
    </div>
  );
}

// Section with Heading
export function Section() {
  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-950 dark:text-white">
          Section Title
        </h2>
        <p className="mt-2 text-sm text-gray-700 dark:text-gray-400">
          Section description goes here.
        </p>
      </div>
      <div className="space-y-4">
        {/* Section content */}
      </div>
    </section>
  );
}

// Two Column Layout
export function TwoColumnLayout() {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="rounded-lg border border-gray-950/10 bg-white p-6 dark:border-white/10 dark:bg-gray-900">
        <h3 className="text-lg font-semibold text-gray-950 dark:text-white">
          Left Column
        </h3>
        <p className="mt-2 text-sm text-gray-700 dark:text-gray-400">
          Content for the left column.
        </p>
      </div>
      <div className="rounded-lg border border-gray-950/10 bg-white p-6 dark:border-white/10 dark:bg-gray-900">
        <h3 className="text-lg font-semibold text-gray-950 dark:text-white">
          Right Column
        </h3>
        <p className="mt-2 text-sm text-gray-700 dark:text-gray-400">
          Content for the right column.
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// INTERACTIVE ELEMENTS
// =============================================================================

// Loading Spinner
export function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-gray-950 border-t-transparent dark:border-gray-400" />
    </div>
  );
}

// Badge
export function Badge() {
  return (
    <span className="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-950 dark:bg-gray-800 dark:text-white">
      New
    </span>
  );
}

// Avatar / Icon Button
export function IconButton() {
  return (
    <button className="inline-flex h-10 w-10 items-center justify-center rounded-full transition-colors hover:bg-gray-100 focus:outline-2 focus:outline-offset-2 focus:outline-blue-500 dark:hover:bg-gray-800">
      <span className="text-gray-950 dark:text-white">👤</span>
    </button>
  );
}

// =============================================================================
// ACCESSIBILITY PATTERNS
// =============================================================================

// Skip Link
export function SkipLink() {
  return (
    <a
      href="#main-content"
      className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-blue-600 focus:px-4 focus:py-2 focus:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
    >
      Skip to main content
    </a>
  );
}

// Screen Reader Only Text
export function ScreenReaderOnly() {
  return (
    <button>
      <span className="sr-only">Close menu</span>
      <span aria-hidden="true">✕</span>
    </button>
  );
}

// =============================================================================
// USAGE NOTES
// =============================================================================

/**
 * How to Use These Patterns:
 *
 * 1. Copy the pattern you need
 * 2. Paste into your component
 * 3. Adjust content but keep classes
 * 4. Test in both light and dark mode
 * 5. Test keyboard navigation
 * 6. Test with screen reader if interactive
 *
 * Remember:
 * - Buttons are ALWAYS rounded-full
 * - Default button bg is gray-950 (not bg-primary)
 * - Default text size is text-sm for prose
 * - All borders use /10 opacity (e.g., border-gray-950/10)
 * - Focus rings are blue-500 with 2px offset
 * - Dark mode uses dark: prefix
 */
