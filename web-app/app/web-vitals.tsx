'use client';

import { useReportWebVitals } from 'next/web-vitals';

export function WebVitals() {
  useReportWebVitals((metric) => {
    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.log(metric);
    }

    // Send to analytics in production
    // Example: Send to Vercel Analytics
    if (typeof window !== 'undefined' && 'va' in window) {
      const va = (window as any).va;
      va('event', {
        name: metric.name,
        value: metric.value,
        label: metric.id,
        rating: metric.rating,
      });
    }

    // You can also send to other analytics services here
    // Example: Google Analytics 4
    // gtag('event', metric.name, {
    //   value: Math.round(metric.name === 'CLS' ? metric.value * 1000 : metric.value),
    //   metric_id: metric.id,
    //   metric_value: metric.value,
    //   metric_delta: metric.delta,
    // });
  });

  return null;
}
