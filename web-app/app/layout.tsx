import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Machine Learning Introduction",
  description: "A comprehensive, hands-on introduction to machine learning that prioritizes understanding through application",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
