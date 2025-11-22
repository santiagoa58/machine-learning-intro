import { filterJoin } from "@/lib/utils";
import { useMemo } from "react";

interface PerformanceEndOptions {
  error?: unknown;
  detail?: object;
  subName?: string;
}

export const createPerformanceMeasure = (name: string) => {
  return {
    start(subName?: string) {
      return performance.mark(`${filterJoin("-", name, subName)}-start`);
    },
    end({ subName, error, detail }: PerformanceEndOptions = {}) {
      const markName = filterJoin("-", name, subName);
      if (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`Error during ${name}:`, message);
        performance.mark(`${markName}-end-error`);
        return performance.measure(name, {
          start: `${markName}-start`,
          end: `${markName}-end-error`,
          detail: { error: message, ...detail },
        });
      }
      performance.mark(`${markName}-end`);
      return performance.measure(name, {
        start: `${markName}-start`,
        end: `${markName}-end`,
        detail: detail,
      });
    },
  } as const;
};

export const usePerformanceMeasure = (name: string) => {
  return useMemo(() => createPerformanceMeasure(name), [name]);
};
