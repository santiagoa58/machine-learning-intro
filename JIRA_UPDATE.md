# JIRA Update - Component Library & Content Architecture

## Critical Changes Required

### Tech Stack Updates

**REMOVE:**
- ❌ MDX for content (not needed, adds complexity)

**ADD:**
- ✅ **shadcn/ui** - Component library (Radix UI + Tailwind)
- ✅ **@tanstack/react-query** - Async state management for Python execution
- ✅ **@monaco-editor/react** - VS Code editor for code blocks
- ✅ **Vercel AI SDK** - For future AI features (chat, code assistance)
- ✅ **Pyodide** - Python in WebAssembly
- ✅ **Structured Content System** - JSON/TypeScript content files

---

## NEW EPIC-10: Component Library & Design System 🎨

**Status:** To Do (CRITICAL)
**Priority:** Critical - Blocking Sprint 1
**Target:** Week 1 of Sprint 1
**Owner:** Tech Lead

**Description:** Set up professional component library using shadcn/ui and establish design system. This is foundational infrastructure that all other features depend on.

**Business Value:**
- Consistent UX across all features
- Faster development with pre-built accessible components
- Professional UI out of the box
- AI-ready with Vercel SDK

### Stories

#### [STORY-36] shadcn/ui Setup & Base Components
**Priority:** Critical
**Story Points:** 8
**Sprint:** Sprint 1 (Week 1)

**Tasks:**
- [TASK-149] ⚪ Initialize shadcn/ui in project (1 point)
  ```bash
  npx shadcn@latest init
  ```
- [TASK-150] ⚪ Install base components: Button, Card, Input, Textarea (2 points)
  ```bash
  npx shadcn@latest add button card input textarea
  ```
- [TASK-151] ⚪ Install interactive components: Tabs, Accordion, Dialog, Tooltip (2 points)
  ```bash
  npx shadcn@latest add tabs accordion dialog tooltip
  ```
- [TASK-152] ⚪ Install form components: Form, Label, Select, Checkbox, RadioGroup (2 points)
  ```bash
  npx shadcn@latest add form label select checkbox radio-group
  ```
- [TASK-153] ⚪ Configure design tokens in tailwind.config.ts (1 point)

**Acceptance Criteria:**
- shadcn/ui components available in `components/ui/`
- All components work in light and dark mode
- Components follow WCAG 2.1 AA accessibility standards
- Design tokens configured for brand colors

#### [STORY-37] Code Editor & Execution Infrastructure
**Priority:** Critical
**Story Points:** 13
**Sprint:** Sprint 1 (Week 1-2)

**Tasks:**
- [TASK-154] ⚪ Install Monaco Editor and React Query (2 points)
  ```bash
  npm install @monaco-editor/react @tanstack/react-query
  ```
- [TASK-155] ⚪ Create CodeEditor component with Monaco (3 points)
  - Python syntax highlighting
  - Dark/light theme support
  - Line numbers and minimap
  - Keyboard shortcuts

- [TASK-156] ⚪ Set up Pyodide integration with React Query (5 points)
  - Lazy load Pyodide (3MB)
  - Cache Python packages (numpy, pandas, matplotlib)
  - Handle loading states
  - Error boundary for execution failures

- [TASK-157] ⚪ Create PythonExecutor component (3 points)
  - CodeEditor + Run button
  - Output console
  - Execution time display
  - Clear output button

**Example Component:**
```tsx
'use client';

import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import Editor from '@monaco-editor/react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2 } from 'lucide-react';

export function PythonExecutor({
  initialCode,
  packages = ['numpy', 'pandas']
}: PythonExecutorProps) {
  const [code, setCode] = useState(initialCode);

  const executeMutation = useMutation({
    mutationFn: async (codeToRun: string) => {
      const pyodide = await loadPyodide();
      await pyodide.loadPackagesFromImports(codeToRun);
      return await pyodide.runPythonAsync(codeToRun);
    },
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Python Code</CardTitle>
      </CardHeader>
      <CardContent>
        <Editor
          height="400px"
          language="python"
          value={code}
          onChange={(value) => setCode(value || '')}
          theme={theme === 'dark' ? 'vs-dark' : 'light'}
        />
        <Button
          onClick={() => executeMutation.mutate(code)}
          disabled={executeMutation.isPending}
        >
          {executeMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Run Code
        </Button>
        {executeMutation.data && (
          <pre className="mt-4 p-4 bg-black text-green-400 rounded font-mono text-sm">
            {executeMutation.data}
          </pre>
        )}
      </CardContent>
    </Card>
  );
}
```

#### [STORY-38] Vercel AI SDK Integration
**Priority:** High
**Story Points:** 5
**Sprint:** Sprint 1 (Week 2)

**Tasks:**
- [TASK-158] ⚪ Install Vercel AI SDK (1 point)
  ```bash
  npm install ai @ai-sdk/openai
  ```
- [TASK-159] ⚪ Set up API route for AI chat (2 points)
- [TASK-160] ⚪ Create ChatInterface component using useChat hook (2 points)

**Future Use Cases:**
- AI tutor chat for questions
- Code explanation and debugging
- Concept clarification
- Personalized hints

#### [STORY-39] Chart & Visualization Library
**Priority:** High
**Story Points:** 5
**Sprint:** Sprint 1 (Week 2)

**Tasks:**
- [TASK-161] ⚪ Install Recharts (React charts library) (1 point)
  ```bash
  npm install recharts
  ```
- [TASK-162] ⚪ Create Chart wrapper components (2 points)
  - LineChart component
  - ScatterPlot component
  - BarChart component

- [TASK-163] ⚪ Integrate with Pyodide matplotlib output (2 points)
  - Convert matplotlib figures to SVG
  - Display in React

**Acceptance Criteria:**
- Python matplotlib code renders in browser
- Interactive charts with tooltips
- Responsive to screen size
- Works with dark mode

---

## NEW EPIC-11: Content Architecture & Management System 📝

**Status:** To Do (CRITICAL)
**Priority:** Critical - Technical Debt
**Target:** Week 2-3 of Sprint 1
**Owner:** Senior Staff Engineer

**Description:** Refactor hardcoded JSX content to structured, maintainable content management system following clean architecture principles.

**Business Value:**
- Content updates don't require code changes
- Non-technical contributors can edit content
- Type-safe content schema
- Single source of truth for all content
- Easy to add new tutorials without touching components

**Current Problem:**
- 1000+ lines of hardcoded JSX in doc pages ❌
- Content mixed with presentation ❌
- Changes require code deployment ❌
- Not scalable ❌
- Violates separation of concerns ❌

**Solution: Structured Content System**

### Architecture Design

```
project/
├── content/                          # Content as data
│   ├── docs/
│   │   ├── readme.json              # Structured content
│   │   ├── guidelines.json
│   │   ├── learning-science.json
│   │   ├── improvement-guide.json
│   │   └── jira.json
│   ├── tutorials/
│   │   ├── linear-regression.json   # Tutorial content
│   │   ├── knn.json
│   │   ├── logistic-regression.json
│   │   └── svm.json
│   ├── metadata.json                # Navigation, routes
│   └── schema.ts                    # TypeScript interfaces
│
├── components/
│   ├── content/                     # Content renderers
│   │   ├── ContentRenderer.tsx      # Generic renderer
│   │   ├── CodeBlock.tsx           # Renders code sections
│   │   ├── MathBlock.tsx           # Renders LaTeX
│   │   ├── Section.tsx             # Renders sections
│   │   └── QuickCheck.tsx          # Renders quizzes
│   └── ui/                         # shadcn components
│       └── ...
│
└── lib/
    ├── content-loader.ts           # Load and parse content
    ├── content-validator.ts        # Validate against schema
    └── types.ts                    # All TypeScript types
```

### Content Schema Example

```typescript
// content/schema.ts
export interface ContentBlock {
  type: 'paragraph' | 'heading' | 'code' | 'list' | 'math' | 'callout' | 'quiz' | 'interactive-code';
  content: string;
  metadata?: {
    level?: 1 | 2 | 3 | 4;      // For headings
    language?: string;             // For code blocks
    executable?: boolean;          // Can code be run?
    id?: string;                   // For anchors
  };
}

export interface ContentSection {
  id: string;
  title: string;
  blocks: ContentBlock[];
  subsections?: ContentSection[];
}

export interface Tutorial {
  slug: string;
  title: string;
  description: string;
  metadata: {
    difficulty: 'beginner' | 'intermediate' | 'advanced';
    duration: number;  // minutes
    prerequisites: string[];
    learningObjectives: string[];
  };
  sections: ContentSection[];
}
```

### Content File Example

```json
// content/tutorials/linear-regression.json
{
  "slug": "linear-regression",
  "title": "Linear Regression: From Application to Theory",
  "description": "Learn linear regression by building a real stock price predictor",
  "metadata": {
    "difficulty": "beginner",
    "duration": 60,
    "prerequisites": ["python-basics", "numpy-basics"],
    "learningObjectives": [
      "Understand what linear regression does",
      "Build and train a linear regression model",
      "Evaluate model performance with R² and MSE"
    ]
  },
  "sections": [
    {
      "id": "introduction",
      "title": "Introduction",
      "blocks": [
        {
          "type": "paragraph",
          "content": "Linear regression is the foundation of machine learning..."
        },
        {
          "type": "heading",
          "content": "What You'll Learn",
          "metadata": { "level": 2 }
        },
        {
          "type": "list",
          "content": ["Build a stock price predictor", "Understand gradient descent", "Evaluate model accuracy"],
          "metadata": { "ordered": false }
        }
      ]
    },
    {
      "id": "application",
      "title": "Application: Stock Price Prediction",
      "blocks": [
        {
          "type": "code",
          "content": "import numpy as np\nfrom sklearn.linear_model import LinearRegression\n\n# Load stock data\nX = ...\ny = ...",
          "metadata": {
            "language": "python",
            "executable": true
          }
        },
        {
          "type": "quiz",
          "content": {
            "question": "What are the two inputs to train_test_split()?",
            "answer": "Features (X) and targets (y)",
            "hint": "Think about what the model needs to learn from"
          }
        }
      ]
    }
  ]
}
```

### Stories

#### [STORY-40] Content Schema & Type Definitions
**Priority:** Critical
**Story Points:** 8
**Sprint:** Sprint 1 (Week 2)

**Tasks:**
- [TASK-164] ⚪ Design content schema (3 points)
  - ContentBlock interface
  - ContentSection interface
  - Tutorial interface
  - DocPage interface
  - Metadata types

- [TASK-165] ⚪ Create TypeScript type definitions in lib/types.ts (2 points)
- [TASK-166] ⚪ Create Zod validation schemas (2 points)
- [TASK-167] ⚪ Write content schema documentation (1 point)

**Acceptance Criteria:**
- All content types defined in TypeScript
- Zod schemas for runtime validation
- Documentation with examples
- Type-safe content loading

#### [STORY-41] Content Loader & Renderer System
**Priority:** Critical
**Story Points:** 13
**Sprint:** Sprint 1 (Week 2-3)

**Tasks:**
- [TASK-168] ⚪ Create ContentLoader utility (3 points)
  - Load JSON files
  - Validate against schema
  - Cache loaded content
  - Error handling

- [TASK-169] ⚪ Create ContentRenderer component (5 points)
  - Switch on block type
  - Render appropriate component
  - Handle nested sections
  - Support all block types

- [TASK-170] ⚪ Create block-specific renderers (3 points)
  - HeadingBlock
  - ParagraphBlock
  - CodeBlock (syntax highlighted)
  - ListBlock
  - MathBlock (KaTeX)
  - CalloutBlock
  - QuizBlock

- [TASK-171] ⚪ Write unit tests for renderers (2 points)

**Example ContentRenderer:**
```tsx
// components/content/ContentRenderer.tsx
import { ContentBlock } from '@/lib/types';
import { HeadingBlock } from './HeadingBlock';
import { ParagraphBlock } from './ParagraphBlock';
import { CodeBlock } from './CodeBlock';
import { QuizBlock } from './QuizBlock';

export function ContentRenderer({ blocks }: { blocks: ContentBlock[] }) {
  return (
    <>
      {blocks.map((block, index) => {
        switch (block.type) {
          case 'heading':
            return <HeadingBlock key={index} block={block} />;
          case 'paragraph':
            return <ParagraphBlock key={index} block={block} />;
          case 'code':
            return <CodeBlock key={index} block={block} />;
          case 'quiz':
            return <QuizBlock key={index} block={block} />;
          case 'interactive-code':
            return <PythonExecutor key={index} code={block.content} />;
          default:
            return null;
        }
      })}
    </>
  );
}
```

#### [STORY-42] Refactor Documentation Pages to Use Structured Content
**Priority:** High
**Story Points:** 13
**Sprint:** Sprint 1 (Week 3)

**Tasks:**
- [TASK-172] ⚪ Convert README.md to readme.json (3 points)
  - Extract all content
  - Structure into blocks
  - Validate schema

- [TASK-173] ⚪ Convert PROJECT_GUIDELINES.md to guidelines.json (3 points)
- [TASK-174] ⚪ Convert LEARNING_SCIENCE_REVIEW.md to learning-science.json (3 points)
- [TASK-175] ⚪ Convert IMPROVEMENT_GUIDE.md to improvement-guide.json (2 points)
- [TASK-176] ⚪ Convert JIRA.md to jira.json (2 points)

**Before:**
```tsx
// ❌ Hardcoded - 1000+ lines
export default function ReadmePage() {
  return (
    <>
      <h1>Machine Learning Introduction</h1>
      <p>A comprehensive...</p>
      {/* 998 more lines of JSX */}
    </>
  );
}
```

**After:**
```tsx
// ✅ Clean - Dynamic rendering
import { getDocContent } from '@/lib/content-loader';
import { ContentRenderer } from '@/components/content/ContentRenderer';

export default async function ReadmePage() {
  const content = await getDocContent('readme');

  return (
    <article>
      <h1>{content.title}</h1>
      <ContentRenderer blocks={content.blocks} />
    </article>
  );
}
```

**Benefits:**
- 1000 lines of JSX → 10 lines of code ✅
- Content separate from code ✅
- Type-safe ✅
- Easy to update ✅
- Reusable renderer ✅

#### [STORY-43] Content Management CLI Tool
**Priority:** Medium
**Story Points:** 8
**Sprint:** Sprint 1 (Week 3)

**Tasks:**
- [TASK-177] ⚪ Create CLI to convert Markdown → JSON (5 points)
  ```bash
  npm run convert -- input.md output.json
  ```
- [TASK-178] ⚪ Create CLI to validate content files (2 points)
  ```bash
  npm run validate-content
  ```
- [TASK-179] ⚪ Add content linting to CI/CD (1 point)

**Benefits:**
- Easy content migration
- Automated validation
- Catch content errors early

---

## Updates to Existing Epics

### EPIC-3: Web Application Infrastructure
**UPDATED:** Remove MDX references, update to use structured content

**Updated Story: [STORY-11] Tutorial Content System**
- ~~Set up MDX integration~~ → Use structured JSON content
- ~~Create tutorial page template~~ → Create ContentRenderer
- Keep: syntax highlighting, LaTeX, component library

### EPIC-5: Content Migration & Enhancement
**UPDATED:** Change from "Convert to MDX" to "Convert to JSON"

**Updated Stories:**
- [STORY-18] Convert Linear Regression to **JSON** (not MDX)
- [STORY-19] Convert KNN to **JSON**
- [STORY-20] Convert Logistic Regression to **JSON**
- [STORY-21] Convert SVM to **JSON**
- [STORY-22] Create Foundation Tutorials in **JSON**

---

## Updated Tech Stack

### Current (Implemented)
- ✅ Next.js 16.0.0
- ✅ React 19.2.0
- ✅ TypeScript 5
- ✅ Tailwind CSS v4
- ✅ Turbopack

### To Add (Sprint 1 Week 1)
- **shadcn/ui** - Component library
- **@tanstack/react-query** - Async state
- **@monaco-editor/react** - Code editor
- **Pyodide** - Python execution
- **Recharts** - Visualizations
- **KaTeX** - Math rendering
- **Zod** - Schema validation

### To Add (Sprint 1 Week 2)
- **Vercel AI SDK** - AI features
- **OpenAI API** - For AI tutor (optional)

### Architecture Pattern
```
Presentation Layer (React Components)
    ↓
Business Logic Layer (lib/ utilities)
    ↓
Data Layer (content/ JSON files)
```

This follows **Clean Architecture** principles:
- ✅ Separation of concerns
- ✅ Dependency inversion
- ✅ Single responsibility
- ✅ DRY (Don't Repeat Yourself)
- ✅ Type safety throughout

---

## Priority Ranking

### Must Do Sprint 1 Week 1 (Blocking Everything)
1. shadcn/ui setup (STORY-36)
2. Monaco Editor + Pyodide (STORY-37)
3. Content schema design (STORY-40)

### Must Do Sprint 1 Week 2
4. Content loader & renderer (STORY-41)
5. Vercel AI SDK setup (STORY-38)
6. Refactor doc pages (STORY-42)

### Should Do Sprint 1 Week 3
7. Chart library (STORY-39)
8. Content CLI tool (STORY-43)

---

## Success Metrics

### Code Quality
- Lines of code reduced by 80% (1000 → 200 lines for docs)
- Component reuse increased
- Type safety 100%
- Zero hardcoded content

### Developer Experience
- New tutorials added in < 1 hour
- Content updates with zero code changes
- AI features ready to integrate
- Python execution working

### User Experience
- Professional UI with shadcn
- Code execution in browser
- Interactive visualizations
- Consistent design system

---

## Questions for Product Owner

1. **AI Features**: Should we prioritize AI tutor chat in Sprint 1 or defer to Sprint 2?
2. **Content Format**: JSON vs YAML vs other? (Recommend JSON for TypeScript support)
3. **Content Editing**: Should we build a web-based content editor or stick with file editing?
4. **Code Execution**: What Python packages are priority? (numpy, pandas, sklearn, matplotlib?)

---

**Created:** 2025-10-25
**Status:** Ready for Review
**Next Steps:** Review with team, approve, start implementation
