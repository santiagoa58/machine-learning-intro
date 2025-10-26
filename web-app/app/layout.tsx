import type { Metadata, Viewport } from "next";
import { clsx } from "clsx";
import { GeistMono } from "geist/font/mono";
import localFont from "next/font/local";
import { WebVitals } from "./web-vitals";
import "./globals.css";

const InterVariable = localFont({
  variable: "--font-inter",
  src: [
    { path: "./InterVariable.woff2", style: "normal" },
    { path: "./InterVariable-Italic.woff2", style: "italic" },
  ],
});

export const metadata: Metadata = {
  title: {
    default: "Machine Learning Introduction",
    template: "%s | ML Introduction",
  },
  description: "A comprehensive, hands-on introduction to machine learning that prioritizes understanding through application. Learn by doing with interactive tutorials.",
  keywords: ["machine learning", "ML tutorial", "Python", "scikit-learn", "data science", "AI education"],
  authors: [{ name: "ML Introduction Team" }],
  creator: "ML Introduction Team",
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://machine-learning-intro.vercel.app",
    title: "Machine Learning Introduction",
    description: "Learn machine learning through hands-on interactive tutorials",
    siteName: "ML Introduction",
  },
  twitter: {
    card: "summary_large_image",
    title: "Machine Learning Introduction",
    description: "Learn machine learning through hands-on interactive tutorials",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0a" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={clsx(
        GeistMono.variable,
        InterVariable.variable,
        "scroll-pt-16 font-sans antialiased dark:bg-gray-950"
      )}
    >
      <body>
        <div className="isolate">
          <a
            href="#main-content"
            className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-blue-600 focus:text-white focus:rounded focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Skip to main content
          </a>
          <main id="main-content">
            {children}
          </main>
          <WebVitals />
        </div>
      </body>
    </html>
  );
}
