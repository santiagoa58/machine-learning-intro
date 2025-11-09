import { useMemo } from "react";

interface PerformanceEndOptions {
  error?: unknown;
  detail?: object;
}

export const usePerformanceMeasure = (name: string) => {
  return useMemo(
    () => ({
      start() {
        return performance.mark(`${name}-start`);
      },
      end({ error, detail }: PerformanceEndOptions = {}) {
        if (error) {
          const message =
            error instanceof Error ? error.message : String(error);
          console.error(`Error during ${name}:`, message);
          performance.mark(`${name}-end-error`);
          return performance.measure(name, {
            start: `${name}-start`,
            end: `${name}-end-error`,
            detail: { error: message, ...detail },
          });
        }
        performance.mark(`${name}-end`);
        return performance.measure(name, {
          start: `${name}-start`,
          end: `${name}-end-error`,
          detail: detail,
        });
      },
    } as const),
    [name]
  );
};
