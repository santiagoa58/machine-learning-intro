# MDX Content System

This project uses MDX for rich, interactive content with React components embedded in markdown.

## Available Components

### Data Visualization

#### Chart
Display beautiful charts using Recharts.

**Usage:**
```tsx
import { Chart } from '@/components/mdx';

<Chart
  type="line"  // or "bar" | "scatter"
  data={[
    { x: 1, y: 10 },
    { x: 2, y: 20 }
  ]}
  xKey="x"
  yKey="y"  // or ["y1", "y2"] for multiple lines
  title="My Chart"
  description="Optional description"
  height={300}
  colors={['#3b82f6', '#10b981']}  // optional
/>
```

#### DataTable
Display data in a beautiful table.

**Usage:**
```tsx
import { DataTable } from '@/components/mdx';

<DataTable
  data={[
    { name: "Alice", score: 95 },
    { name: "Bob", score: 87 }
  ]}
  columns={['name', 'score']}  // optional, defaults to all keys
  title="Student Scores"
  description="Exam results"
  caption="Fall 2024 Semester"
/>
```

### Interactive Elements

#### CodePlayground
Embeddable code editor with live preview using Sandpack.

**Usage:**
```tsx
import { CodePlayground } from '@/components/mdx';

<CodePlayground
  code={`console.log('Hello World!')`}
  title="JavaScript Example"
  template="vanilla"  // or "react" | "react-ts" | "vue" | "angular"
  editorHeight={400}
/>

// Multiple files
<CodePlayground
  files={{
    '/App.js': 'export default function App() { return <h1>Hello</h1> }',
    '/index.js': 'import App from "./App"; ...'
  }}
  template="react"
/>
```

### Content Enhancement

#### Callout
Highlight important information with styled callouts.

**Usage:**
```tsx
import { Callout } from '@/components/mdx';

<Callout type="info" title="Key Concept">
This is important information for the reader.
</Callout>

// Types: "info" | "warning" | "success" | "error" | "tip"
```

#### Steps
Create step-by-step guides.

**Usage:**
```tsx
import { Steps, Step } from '@/components/mdx';

<Steps>
  <Step title="Install Dependencies">
    Run `npm install` to install all required packages.
  </Step>

  <Step title="Configure Settings">
    Update your config file with the API keys.
  </Step>
</Steps>
```

#### Tabs
Organize content in tabs.

**Usage:**
```tsx
import { Tabs } from '@/components/mdx';

<Tabs
  items={[
    {
      label: 'JavaScript',
      content: <div>JS content here</div>
    },
    {
      label: 'TypeScript',
      content: <div>TS content here</div>
    }
  ]}
/>
```

## Writing MDX Content

### Basic Structure

Create `.mdx` files in `content/algorithms/`:

```mdx
---
title: Your Algorithm Name
description: Brief description
prerequisites:
  - Item 1
  - Item 2
learningOutcomes:
  - Outcome 1
  - Outcome 2
---

## Introduction

Your content here...

### Using Components

Import and use React components directly:

```mdx
import { Callout, Chart } from '@/components/mdx';

<Callout type="tip" title="Pro Tip">
Remember to check your assumptions!
</Callout>
```

## Standard Markdown Features

All standard markdown features work as expected:

- **Bold text**
- *Italic text*
- `Inline code`
- [Links](https://example.com)
- Lists (ordered and unordered)
- Tables
- Code blocks with syntax highlighting
- Blockquotes
- Headers with anchor links (h2 and h3)

### Code Blocks

Use fenced code blocks with language specifiers for syntax highlighting:

\`\`\`python
def hello():
    print("Hello, World!")
\`\`\`

\`\`\`javascript
function hello() {
  console.log("Hello, World!");
}
\`\`\`

### Tables

Standard markdown tables are automatically styled:

```markdown
| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
```

### Blockquotes

Use `>` for blockquotes:

```markdown
> This is an important note or quote.
```

## Tips for Writing Content

1. **Keep it Simple**: Start with standard markdown and add components only when needed
2. **Use Callouts**: Highlight key concepts, warnings, and tips
3. **Code Examples**: Always include runnable code examples
4. **Visual Aids**: Use charts and tables to illustrate concepts
5. **Progressive Learning**: Build from simple to complex concepts
6. **Interactive**: Use CodePlayground for hands-on learning

## Performance Notes

- All components are code-split automatically
- MDX is compiled at build time for optimal performance
- Client-side rendering for dynamic components
- Server-side rendering for static content

## Troubleshooting

### Component Not Found

Make sure you import components at the top of your MDX file:

```mdx
import { Callout } from '@/components/mdx';
```

### Complex JavaScript Expressions

`next-mdx-remote` has limitations with complex JavaScript expressions in component props. For complex data:

1. Keep data structures simple
2. Use separate component files for complex logic
3. Consider using standard markdown tables instead of DataTable for simple data

### Styling Issues

All components use Tailwind CSS and support dark mode automatically. Custom styles can be added via className prop where supported.
