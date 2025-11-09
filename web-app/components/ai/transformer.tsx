import { usePerformanceMeasure } from "@/app/hooks/usePerformanceMeasure";
import { pipeline, SummarizationPipeline } from "@huggingface/transformers";
import { useCallback, useEffect, useState } from "react";

export const createSummarizer = async () => {
  let webGPUAvailable = true;
  if (!("gpu" in navigator && navigator.gpu)) {
    console.warn("WebGPU not supported on this browser.");
    webGPUAvailable = false;
  }
  const summarizer = await pipeline("summarization", "google/t5-small", {
    device: webGPUAvailable ? "webgpu" : undefined,
  });
  return summarizer;
};

export const useSummarizer = () => {
  const summarizerInitPerformance = usePerformanceMeasure(
    "Summarizer Initialization"
  );
  const summarizerPerformance = usePerformanceMeasure("Summarizer Execution");
  const [summarizer, setSummarizer] = useState<SummarizationPipeline | null>(
    null
  );

  useEffect(() => {
    (async () => {
      if (summarizer) {
        return;
      }
      try {
        summarizerInitPerformance.start();
        const pipeline = await createSummarizer();
        setSummarizer(pipeline);
        summarizerInitPerformance.end();
      } catch (error: unknown) {
        summarizerInitPerformance.end({ error });
      }
    })();
  }, [summarizer, summarizerInitPerformance]);

  return useCallback(
    async (text: string) => {
      if (!summarizer) {
        console.warn("Summarizer not initialized yet.");
        return null;
      }
      try {
        summarizerPerformance.start();
        const summaries = await summarizer(text);
        const summaryTexts = summaries
          .flat()
          .map((result) => result?.summary_text)
          .filter(Boolean);
        if (summaryTexts.length === 0) {
          summarizerPerformance.end({ error: "No valid summaries generated" });
          return null;
        }
        summarizerPerformance.end();
        return summaryTexts;
      } catch (error: unknown) {
        summarizerPerformance.end({ error });
        return null;
      }
    },
    [summarizer, summarizerPerformance]
  );
};
