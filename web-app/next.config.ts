import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable React strict mode for better debugging
  reactStrictMode: true,

  // Remove X-Powered-By header
  poweredByHeader: false,

  // Enable compression
  compress: true,

  // Enforce type checking during builds
  typescript: {
    ignoreBuildErrors: false,
  },

  // Experimental features for better performance
  experimental: {
    // Optimize package imports for faster bundling
    optimizePackageImports: ['@headlessui/react', 'lucide-react'],
  },

  // Security and performance headers
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          // Security headers
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload',
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
          {
            key: 'Content-Security-Policy',
            value: [
              "default-src 'self'",
              // Note: In production, replace 'unsafe-eval' and 'unsafe-inline' with nonce-based CSP
              // Development mode requires 'unsafe-eval' for hot module replacement and 'unsafe-inline' for inline scripts
              `script-src 'self' ${process.env.NODE_ENV === 'development' ? "'unsafe-eval' 'unsafe-inline'" : ""}`,
              // Allow inline styles for Next.js fonts, CSS-in-JS, and development tools
              `style-src 'self' ${process.env.NODE_ENV === 'development' ? "'unsafe-inline'" : ""}`,
              "img-src 'self' data: https:",
              // Allow fonts from CDNs and data URIs
              "font-src 'self' data: https:",
              "connect-src 'self' ws: wss:",
              "frame-ancestors 'none'",
            ].join('; '),
          },
        ],
      },
    ];
  },

  // Image optimization configuration
  images: {
    formats: ['image/avif', 'image/webp'],
    remotePatterns: [],
  },
};

export default nextConfig;
