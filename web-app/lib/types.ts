import type { ReactNode } from 'react';

// Layout Props
export interface LayoutProps {
  children: ReactNode;
}

export interface PageProps {
  params: Record<string, string>;
  searchParams: Record<string, string | string[] | undefined>;
}

// Error Props
export interface ErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

// Link Props
export interface DocLinkProps {
  href: string;
  children: ReactNode;
  className?: string;
}

// Algorithm Types
export interface Algorithm {
  id: string;
  name: string;
  description: string;
  category: 'regression' | 'classification';
}

// Feature Types
export interface Feature {
  id: string;
  icon: string;
  title: string;
  description: string;
}

// Documentation Types
export interface DocPage {
  slug: string;
  title: string;
  description: string;
  href: string;
}

// Navigation Types (Compass-style)
export interface Lesson {
  id: string;
  title: string;
  description?: string;
}

export interface Module {
  id: string;
  title: string;
  description?: string;
  lessons: Lesson[];
}

// Constants
export const DOC_PAGES: ReadonlyArray<DocPage> = [
  {
    slug: 'readme',
    title: 'Project Overview & Setup',
    description: 'Learn how to get started with our comprehensive machine learning tutorial platform.',
    href: '/docs/readme',
  },
  {
    slug: 'guidelines',
    title: 'Project Guidelines',
    description: 'Teaching philosophy, content standards, and learning science principles.',
    href: '/docs/guidelines',
  },
  {
    slug: 'learning-science',
    title: 'Learning Science Review',
    description: 'Expert analysis from cognitive psychology and learning science perspectives.',
    href: '/docs/learning-science',
  },
  {
    slug: 'improvement-guide',
    title: 'Improvement Guide',
    description: 'Practical templates for implementing active learning principles.',
    href: '/docs/improvement-guide',
  },
  {
    slug: 'jira',
    title: 'Task Tracker (JIRA)',
    description: 'Project roadmap, sprint plans, and task tracking.',
    href: '/docs/jira',
  },
] as const;

export const ALGORITHMS: ReadonlyArray<Algorithm> = [
  {
    id: 'linear-regression',
    name: 'Linear Regression',
    description: 'Predict continuous values',
    category: 'regression',
  },
  {
    id: 'knn',
    name: 'K-Nearest Neighbors',
    description: 'Classify based on proximity',
    category: 'classification',
  },
  {
    id: 'logistic-regression',
    name: 'Logistic Regression',
    description: 'Binary classification',
    category: 'classification',
  },
  {
    id: 'svm',
    name: 'Support Vector Machines',
    description: 'Optimal decision boundaries',
    category: 'classification',
  },
] as const;

export const FEATURES: ReadonlyArray<Feature> = [
  {
    id: 'active-learning',
    icon: '🧠',
    title: 'Active Learning',
    description: 'Retrieval practice, completion problems, and interactive quizzes',
  },
  {
    id: 'code-execution',
    icon: '💻',
    title: 'Code Execution',
    description: 'Run Python code directly in your browser with Pyodide',
  },
  {
    id: 'progress-tracking',
    icon: '📊',
    title: 'Progress Tracking',
    description: 'Track your learning journey and spaced repetition schedule',
  },
] as const;

// Navigation Modules (Compass-style sidebar structure)
export const NAVIGATION_MODULES: ReadonlyArray<Module> = [
  {
    id: 'supervised-learning',
    title: 'Supervised Learning',
    lessons: ALGORITHMS.map(algo => ({
      id: `/learn/${algo.id}`,
      title: algo.name,
      description: algo.description,
    })),
  },
  {
    id: 'reference',
    title: 'Reference',
    description: 'Guidelines and documentation',
    lessons: DOC_PAGES.map(page => ({
      id: page.href,
      title: page.title,
      description: page.description,
    })),
  },
] as const;
