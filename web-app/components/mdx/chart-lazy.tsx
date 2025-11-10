"use client";

import { Suspense, lazy } from "react";

// Lazy load the heavy Chart component
const Chart = lazy(() => import("./chart").then(m => ({ default: m.Chart })));

interface ChartProps {
  data: Array<Record<string, string | number>>;
  type?: "line" | "bar" | "scatter";
  xKey: string;
  yKey: string | string[];
  title?: string;
  description?: string;
  height?: number;
  colors?: string[];
}

function ChartSkeleton({ height = 300 }: { height?: number }) {
  return (
    <div
      className="my-6 animate-pulse bg-gray-100 dark:bg-gray-800 rounded-lg"
      style={{ height }}
    >
      <div className="h-full flex items-center justify-center text-gray-400 dark:text-gray-600">
        Loading chart...
      </div>
    </div>
  );
}

export function ChartLazy(props: ChartProps) {
  return (
    <Suspense fallback={<ChartSkeleton height={props.height} />}>
      <Chart {...props} />
    </Suspense>
  );
}
