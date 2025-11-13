# Machine Learning Learning Path - Design System

**Version:** 1.0.0
**Last Updated:** 2025-10-26
**Based on:** [Compass Tailwind Template](https://tailwindcss.com/templates/compass)

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Color System](#color-system)
4. [Typography](#typography)
5. [Spacing & Layout](#spacing--layout)
6. [Components](#components)
7. [Interactions & States](#interactions--states)
8. [Accessibility](#accessibility)
9. [Dark Mode](#dark-mode)
10. [Code Examples](#code-examples)

---

## Overview

Our design system is built on **Tailwind CSS v4** with a clean, minimal aesthetic inspired by the Compass template. We prioritize readability, accessibility, and a professional appearance suitable for educational content.

### Tech Stack

- **Tailwind CSS:** v4.x (CSS-first configuration)
- **Framework:** Next.js 16 (React 19)
- **Fonts:** Inter Variable + Geist Mono
- **Testing:** Vitest + React Testing Library

### Design Philosophy

1. **Content First:** Design supports learning, not distracts from it
2. **Accessibility:** WCAG 2.1 AA compliance minimum
3. **Performance:** Minimal CSS, optimized fonts, fast loading
4. **Consistency:** Predictable patterns, reusable components
5. **Clarity:** High contrast, readable text, clear hierarchy

---

## Design Principles

### 1. Clarity Over Cleverness
- Use straightforward layouts
- Avoid unnecessary animations
- Prioritize readability

### 2. Consistency Over Customization
- Follow established patterns
- Reuse components
- Maintain visual rhythm

### 3. Accessibility Is Non-Negotiable
- Semantic HTML always
- Keyboard navigation support
- Screen reader friendly
- High contrast ratios

### 4. Progressive Enhancement
- Core content works without JS
- Enhanced experience with JS
- Graceful degradation

---

## Color System

### Gray Scale (Primary Palette)

Our design is **gray-centric** for a clean, professional look.

```css
/* Light Mode */
--color-white: #ffffff       /* Pure white */
--color-gray-50: #f9fafb     /* Lightest gray */
--color-gray-100: #f3f4f6    /* Very light gray */
--color-gray-200: #e5e7eb    /* Light gray */
--color-gray-300: #d1d5db    /* Medium light gray */
--color-gray-400: #9ca3af    /* Medium gray */
--color-gray-500: #6b7280    /* Medium dark gray */
--color-gray-600: #4b5563    /* Dark gray */
--color-gray-700: #374151    /* Very dark gray */
--color-gray-800: #1f2937    /* Nearly black */
--color-gray-900: #111827    /* Almost black */
--color-gray-950: #030712    /* Darkest (backgrounds) */

/* Dark Mode */
--color-black: #000000       /* Pure black */
```

### Usage Guidelines

| Element | Light Mode | Dark Mode |
|---------|-----------|-----------|
| Page Background | `white` or `gray-50` | `gray-950` |
| Card Background | `white` | `gray-900` |
| Primary Text | `gray-950` | `white` |
| Secondary Text | `gray-700` | `gray-400` |
| Muted Text | `gray-600` | `gray-500` |
| Borders | `gray-950/10` | `white/10` |
| Hover Background | `gray-100` | `gray-800` |

### Accent Colors

Used sparingly for emphasis:

```css
/* Blue - Interactive elements, links, focus states */
--color-blue-500: #3b82f6   /* Primary blue */
--color-blue-600: #2563eb   /* Darker blue */
--color-blue-700: #1d4ed8   /* Darkest blue */

/* Red - Destructive actions, errors */
--color-red-600: #dc2626    /* Primary red */
--color-red-700: #b91c1c    /* Darker red */

/* Green - Success states */
--color-green-600: #16a34a  /* Primary green */

/* Yellow - Warnings */
--color-yellow-500: #eab308 /* Primary yellow */
```

### Color Contrast Requirements

| Combination | Minimum Ratio | Usage |
|------------|---------------|-------|
| gray-950 on white | 21:1 | Body text (light mode) |
| white on gray-950 | 21:1 | Body text (dark mode) |
| gray-700 on white | 10.4:1 | Secondary text (light) |
| gray-400 on gray-950 | 9.7:1 | Secondary text (dark) |
| blue-500 focus rings | 4.5:1 | Interactive states |

---

## Typography

### Font Families

#### Sans-Serif (Body & UI)
**Inter Variable**
- Source: Local `.woff2` files
- Weights: 400 (Regular), 500 (Medium), 600 (Semibold), 700 (Bold)
- Features: `cv11` (stylistic alternates for better readability)
- Usage: All text except code

```css
font-family: var(--font-inter), system-ui, sans-serif;
```

#### Monospace (Code)
**Geist Mono**
- Source: `geist` npm package
- Weights: 400 (Regular), 500 (Medium)
- Usage: Code blocks, inline code, technical content

```css
font-family: var(--font-geist-mono), monospace;
```

### Type Scale

Precise scale with calculated line-heights for optimal readability:

| Class | Size | Line Height | Usage |
|-------|------|-------------|-------|
| `text-xs` | 12px | 16px (1.33) | Fine print, labels |
| `text-sm` | 14px | 20px (1.43) | Body text, prose |
| `text-base` | 16px | 24px (1.5) | Default body |
| `text-lg` | 18px | 28px (1.56) | Subheadings (h3) |
| `text-xl` | 20px | 28px (1.4) | Subheadings (h2) |
| `text-2xl` | 24px | 32px (1.33) | Section titles |
| `text-3xl` | 32px | 40px (1.25) | Page titles (h1) |
| `text-4xl` | 40px | 48px (1.2) | Hero text |
| `text-5xl` | 48px | 48px (1.0) | Display |
| `text-6xl` | 60px | 60px (1.0) | Large display |
| `text-7xl` | 72px | 72px (1.0) | Extra large |
| `text-8xl` | 96px | 96px (1.0) | Huge |
| `text-9xl` | 128px | 128px (1.0) | Massive |

### Typography Hierarchy

#### Headings

```css
/* H1 - Page Titles */
font-size: 32px (text-3xl)
line-height: 40px
font-weight: 700 (bold) or 800 (extrabold)
letter-spacing: -0.025em (tight)
color: gray-950 / white (dark)

/* H2 - Section Titles */
font-size: 20px (text-xl)
line-height: 28px
font-weight: 500 (medium) or 600 (semibold)
letter-spacing: -0.025em
margin-top: 60px (spacing-15)

/* H3 - Subsection Titles */
font-size: 18px (text-lg)
line-height: 28px
font-weight: 500 (medium) or 600 (semibold)
letter-spacing: -0.025em
margin-top: 40px (spacing-10)
```

#### Body Text

```css
/* Default Prose */
font-size: 14px (text-sm)
line-height: 28px (spacing-7)
color: gray-700 / gray-400 (dark)

/* Large Body */
font-size: 16px (text-base)
line-height: 24px
```

### Font Weights

| Weight | Value | Usage |
|--------|-------|-------|
| Regular | 400 | Body text |
| Medium | 500 | Headings (h2, h3), emphasis |
| Semibold | 600 | Strong emphasis, buttons |
| Bold | 700 | H1, very strong emphasis |
| Extrabold | 800 | Optional for H1 |

### Letter Spacing

- **Headings:** `-0.025em` (tight, improves readability at large sizes)
- **Body:** Default (0)
- **All Caps:** `0.05em` (loose, improves legibility)

---

## Spacing & Layout

### Spacing Scale

Tailwind's default 4px base unit:

| Class | Value | Usage |
|-------|-------|-------|
| `spacing-0` | 0px | Reset |
| `spacing-1` | 4px | Tight spacing |
| `spacing-2` | 8px | Compact |
| `spacing-3` | 12px | Default gap |
| `spacing-4` | 16px | Standard |
| `spacing-5` | 20px | Comfortable |
| `spacing-6` | 24px | Spacious |
| `spacing-7` | 28px | Large |
| `spacing-8` | 32px | Extra large |
| `spacing-10` | 40px | Section spacing |
| `spacing-12` | 48px | Major spacing |
| `spacing-15` | 60px | Section breaks |
| `spacing-16` | 64px | Large breaks |

### Layout Patterns

#### Container Widths

```css
/* Sidebar Width */
w-2xs: 256px (16rem)

/* Mobile Sidebar */
w-xs: 320px (20rem)

/* Content Container */
max-w-none: No max width (with prose)
max-w-prose: ~65ch (optimal reading width)
max-w-4xl: 896px (large content)
```

#### Breakpoints

```css
sm: 640px   /* Mobile landscape */
md: 768px   /* Tablet */
lg: 1024px  /* Desktop */
xl: 1280px  /* Large desktop (sidebar toggle) */
2xl: 1536px /* Extra large */
```

### Border Radius

| Class | Value | Usage |
|-------|-------|-------|
| `rounded-sm` | 2px | Tight corners |
| `rounded` | 4px | Default |
| `rounded-md` | 6px | Cards, inputs |
| `rounded-lg` | 8px | Larger cards, modals |
| `rounded-xl` | 12px | Hero sections |
| `rounded-2xl` | 16px | Large sections |
| `rounded-full` | 9999px | Buttons, pills, avatars |

---

## Components

### Buttons

#### Primary Button (Default)

```tsx
<button className="
  inline-flex items-center justify-center gap-2
  rounded-full
  px-3.5 py-2
  text-sm/6 font-semibold
  bg-gray-950 text-white
  hover:bg-gray-800
  focus:outline-2 focus:outline-offset-2 focus:outline-blue-500
  disabled:opacity-50 disabled:pointer-events-none
  dark:bg-gray-700 dark:hover:bg-gray-600
">
  Primary Action
</button>
```

**Key Characteristics:**
- `rounded-full` (pill shape, not `rounded-md`)
- `bg-gray-950` (not `bg-primary`)
- `font-semibold` (600 weight)
- `text-sm/6` (14px with 24px line-height)
- Focus ring with offset
- Icon spacing: `gap-2`

#### Button Variants

**Destructive:**
```tsx
className="... bg-red-600 text-white hover:bg-red-700"
```

**Outline:**
```tsx
className="... border border-gray-950/10 bg-white hover:bg-gray-50
  dark:border-white/10 dark:bg-gray-950"
```

**Secondary:**
```tsx
className="... bg-gray-100 text-gray-950 hover:bg-gray-200
  dark:bg-gray-800 dark:text-white"
```

**Ghost:**
```tsx
className="... hover:bg-gray-100 dark:hover:bg-gray-800"
```

**Link:**
```tsx
className="... text-gray-950 underline-offset-4 hover:underline"
```

#### Button Sizes

```tsx
/* Small */
className="... px-3 py-1.5 text-xs"

/* Default */
className="... px-3.5 py-2 text-sm/6"

/* Large */
className="... px-4 py-2.5 text-sm/6"

/* Icon Only */
className="... h-10 w-10"
```

### Cards

```tsx
<div className="
  rounded-lg
  border border-gray-950/10
  bg-white
  p-6
  dark:border-white/10 dark:bg-gray-900
">
  Card content
</div>
```

### Inputs

```tsx
<input className="
  rounded-md
  border border-gray-950/10
  bg-white
  px-3 py-2
  text-sm
  placeholder:text-gray-400
  focus:outline-2 focus:outline-offset-2 focus:outline-blue-500
  dark:border-white/10 dark:bg-gray-900
" />
```

### Navigation

#### Sidebar Navigation

```tsx
<nav>
  <ul className="
    border-l border-gray-950/10
    dark:border-white/10
  ">
    <li className="
      -ml-px
      border-l border-transparent
      pl-4
      hover:border-gray-400 hover:text-gray-950
      aria-[current=page]:border-gray-950 aria-[current=page]:font-medium
      dark:hover:border-gray-600
      dark:aria-[current=page]:border-white
    ">
      <a href="/lesson">Lesson Title</a>
    </li>
  </ul>
</nav>
```

**Key Characteristics:**
- Left border for visual hierarchy
- Active state: solid border + medium weight
- Hover: border opacity changes
- `aria-current="page"` for active

### Prose Content

Content rendered with `.prose` class gets automatic styling:

```tsx
<article className="prose max-w-none">
  <!-- Markdown or HTML content -->
</article>
```

**Prose Styles:**
- H1: 32px, tight tracking, bold
- H2: 20px, medium weight, 60px top margin
- H3: 18px, medium weight, 40px top margin
- Body: 14px (text-sm), gray-700
- Links: semibold, underline with offset
- Code: backticks, gray background
- Pre: rounded-lg, dark background

---

## Interactions & States

### Focus States

**Visual indicator for keyboard navigation:**

```css
focus:outline-2
focus:outline-offset-2
focus:outline-blue-500
```

- 2px blue outline
- 2px offset from element
- Visible on keyboard focus only (`:focus-visible`)

### Hover States

**Subtle feedback on interactive elements:**

```css
/* Buttons */
hover:bg-gray-800

/* Links */
hover:underline
hover:text-decoration-color-gray-950/50

/* Cards */
hover:shadow-lg
hover:border-gray-950/20
```

### Disabled States

```css
disabled:opacity-50
disabled:pointer-events-none
aria-disabled:opacity-50
```

### Active States

```css
active:scale-95
aria-current=page:font-medium
aria-current=page:border-gray-950
```

### Transitions

**Default transition for most interactive elements:**

```css
transition-colors
duration-150
```

**Smooth scroll:**

```css
scroll-behavior: smooth
scroll-pt-16 /* Offset for fixed headers */
```

---

## Accessibility

### Semantic HTML

Always use appropriate HTML elements:

```html
<!-- ✅ Good -->
<button>Click me</button>
<nav><a href="/page">Link</a></nav>
<article><h1>Title</h1></article>

<!-- ❌ Bad -->
<div onclick="...">Click me</div>
<div><span>Link</span></div>
<div><div>Title</div></div>
```

### ARIA Labels

```tsx
/* Navigation with label */
<nav aria-label="Main navigation">

/* Button with descriptive label */
<button aria-label="Close modal">
  <XIcon />
</button>

/* Current page indicator */
<a href="/current" aria-current="page">

/* Disabled state */
<button aria-disabled="true" disabled>
```

### Skip Links

**Always include for keyboard users:**

```tsx
<a href="#main-content" className="sr-only focus:not-sr-only">
  Skip to main content
</a>

<main id="main-content">
  <!-- Content -->
</main>
```

### Screen Reader Only Content

```css
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}
```

### Color Contrast

**Minimum ratios (WCAG 2.1 AA):**
- Normal text: 4.5:1
- Large text (18px+): 3:1
- Interactive elements: 3:1

**Our ratios exceed minimums:**
- Body text: 10.4:1 (light), 9.7:1 (dark)
- Headings: 21:1 (both modes)

### Keyboard Navigation

**All interactive elements must be:**
- Reachable via Tab key
- Activatable via Enter/Space
- Escapable via Esc (for modals)
- Navigable with arrow keys (for menus)

### High Contrast Mode

```css
@media (prefers-contrast: high) {
  *:focus-visible {
    outline-width: 3px;
    outline-color: currentColor;
  }
}
```

### Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

---

## Dark Mode

### Approach

**Class-based dark mode** (not media query):

```tsx
<html className="dark">
```

Toggle via JavaScript/user preference.

### Color Mappings

| Element | Light Mode | Dark Mode |
|---------|-----------|-----------|
| Page bg | `white` | `bg-gray-950` |
| Card bg | `white` | `bg-gray-900` |
| Text (primary) | `text-gray-950` | `text-white` |
| Text (secondary) | `text-gray-700` | `text-gray-400` |
| Text (muted) | `text-gray-600` | `text-gray-500` |
| Border | `border-gray-950/10` | `border-white/10` |
| Hover bg | `hover:bg-gray-100` | `hover:bg-gray-800` |
| Button bg | `bg-gray-950` | `bg-gray-700` |
| Button hover | `hover:bg-gray-800` | `hover:bg-gray-600` |

### Dark Mode Classes

```css
/* Text */
dark:text-white
dark:text-gray-400

/* Backgrounds */
dark:bg-gray-950
dark:bg-gray-900

/* Borders */
dark:border-white/10

/* Hover */
dark:hover:bg-gray-800
```

### Prose in Dark Mode

The `.prose` class automatically handles dark mode:

```css
.prose {
  /* Light mode: gray-700 */
  color: var(--color-gray-700);

  @variant dark {
    /* Dark mode: gray-400 */
    color: var(--color-gray-400);
  }
}
```

---

## Code Examples

### Complete Button Component

```tsx
import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-full text-sm/6 font-semibold transition-colors focus:outline-2 focus:outline-offset-2 focus:outline-blue-500 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'bg-gray-950 text-white hover:bg-gray-800 dark:bg-gray-700 dark:hover:bg-gray-600',
        destructive: 'bg-red-600 text-white hover:bg-red-700',
        outline: 'border border-gray-950/10 bg-white hover:bg-gray-50 dark:border-white/10 dark:bg-gray-950',
        secondary: 'bg-gray-100 text-gray-950 hover:bg-gray-200 dark:bg-gray-800 dark:text-white',
        ghost: 'hover:bg-gray-100 dark:hover:bg-gray-800',
        link: 'text-gray-950 underline-offset-4 hover:underline dark:text-white',
      },
      size: {
        default: 'px-3.5 py-2',
        sm: 'px-3 py-1.5 text-xs',
        lg: 'px-4 py-2.5',
        icon: 'h-10 w-10',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button';
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';

export { Button, buttonVariants };
```

### Prose Content Rendering

```tsx
import ReactMarkdown from 'react-markdown';

export function ContentRenderer({ content }) {
  return (
    <article className="prose max-w-none space-y-6">
      <div className="text-section">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    </article>
  );
}
```

### Layout with Fonts

```tsx
import { clsx } from 'clsx';
import { GeistMono } from 'geist/font/mono';
import localFont from 'next/font/local';
import './globals.css';

const InterVariable = localFont({
  variable: '--font-inter',
  src: [
    { path: './InterVariable.woff2', style: 'normal' },
    { path: './InterVariable-Italic.woff2', style: 'italic' },
  ],
});

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      className={clsx(
        GeistMono.variable,
        InterVariable.variable,
        'scroll-pt-16 font-sans antialiased dark:bg-gray-950'
      )}
    >
      <body>
        <div className="isolate">{children}</div>
      </body>
    </html>
  );
}
```

---

## Quick Reference

### Color Classes (Most Used)

```css
/* Backgrounds */
bg-white
bg-gray-50
bg-gray-100
bg-gray-950
dark:bg-gray-900

/* Text */
text-gray-950
text-gray-700
text-gray-600
dark:text-white
dark:text-gray-400

/* Borders */
border-gray-950/10
dark:border-white/10
```

### Typography Classes

```css
text-sm        /* 14px - Default body */
text-base      /* 16px - Large body */
text-lg        /* 18px - H3 */
text-xl        /* 20px - H2 */
text-3xl       /* 32px - H1 */

font-medium    /* 500 - H2, H3 */
font-semibold  /* 600 - Buttons, emphasis */
font-bold      /* 700 - H1 */
```

### Spacing Classes

```css
p-6           /* Padding: 24px */
px-3.5        /* Horizontal: 14px */
py-2          /* Vertical: 8px */
space-y-6     /* Vertical gap: 24px */
gap-2         /* Flex gap: 8px */
```

### Border Radius

```css
rounded-md    /* 6px - Cards, inputs */
rounded-lg    /* 8px - Larger cards */
rounded-full  /* 9999px - Buttons */
```

---

## Resources

- **Compass Template:** `/theme-guide/compass-reference/`
- **Our Implementation:** `/web-app/`
- **Tailwind CSS v4 Docs:** https://tailwindcss.com/docs
- **Inter Font:** https://rsms.me/inter/
- **Geist Font:** https://vercel.com/font
- **WCAG Guidelines:** https://www.w3.org/WAI/WCAG21/quickref/

---

## Changelog

### Version 1.0.0 (2025-10-26)
- Initial design system documentation
- Migrated to Tailwind CSS v4
- Adopted Compass design system
- Added Inter Variable + Geist Mono fonts
- Implemented gray-centric color palette
- Created comprehensive component library

---

**Maintained by:** ML Learning Path Team
**Questions?** Reference `/theme-guide/compass-reference/` for original Compass patterns.
