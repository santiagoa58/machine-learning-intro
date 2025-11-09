import type { ReactNode } from "react";

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
  category: "regression" | "classification";
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
