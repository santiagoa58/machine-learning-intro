"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Sandpack } from "@codesandbox/sandpack-react";
import { ComponentProps } from "react";

export type CodeTemplate = NonNullable<
  ComponentProps<typeof Sandpack>["template"]
>;

const CODE_TEMPLATES: Readonly<Record<CodeTemplate, CodeTemplate>> = {
  static: "static",
  angular: "angular",
  react: "react",
  "react-ts": "react-ts",
  solid: "solid",
  svelte: "svelte",
  "test-ts": "test-ts",
  "vanilla-ts": "vanilla-ts",
  vanilla: "vanilla",
  vue: "vue",
  "vue-ts": "vue-ts",
  node: "node",
  nextjs: "nextjs",
  vite: "vite",
  "vite-react": "vite-react",
  "vite-react-ts": "vite-react-ts",
  "vite-preact": "vite-preact",
  "vite-preact-ts": "vite-preact-ts",
  "vite-vue": "vite-vue",
  "vite-vue-ts": "vite-vue-ts",
  "vite-svelte": "vite-svelte",
  "vite-svelte-ts": "vite-svelte-ts",
  astro: "astro",
};

const LANGUAGE_MAPPING = {
  ...CODE_TEMPLATES,
  javascript: "vanilla",
  js: "vanilla",
  ts: "vanilla-ts",
  typescript: "vanilla-ts",
} as const;

type CodeLanguage = keyof typeof LANGUAGE_MAPPING;

export const isCodePlaygroundSupported = (
  language: string
): language is CodeLanguage => {
  language = language?.toLowerCase();
  return Boolean(language && LANGUAGE_MAPPING[language as never]);
};
interface CodePlaygroundProps {
  code?: string;
  files?: Record<string, string>;
  title?: string;
  description?: string;
  language?: CodeLanguage;
  showLineNumbers?: boolean;
  editorHeight?: number;
}

export function CodePlayground({
  code,
  files,
  title,
  description,
  language = "vanilla",
  showLineNumbers = true,
  editorHeight = 400,
}: CodePlaygroundProps) {
  // Build the files object
  const sandpackFiles = files || (code ? { "/index.js": code } : undefined);

  const playground = (
    <Sandpack
      template={LANGUAGE_MAPPING[language]}
      files={sandpackFiles}
      theme="auto"
      options={{
        showLineNumbers,
        editorHeight,
        showNavigator: false,
        showTabs: Object.keys(sandpackFiles || {}).length > 1,
        closableTabs: false,
      }}
    />
  );

  if (title || description) {
    return (
      <Card className="my-6 overflow-hidden" title={`${title} code playground`}>
        {(title || description) && (
          <CardHeader>
            {title && <CardTitle>{title}</CardTitle>}
            {description && <CardDescription>{description}</CardDescription>}
          </CardHeader>
        )}
        <CardContent className="p-0">{playground}</CardContent>
      </Card>
    );
  }

  return <div className="my-6">{playground}</div>;
}
