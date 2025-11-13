# Theme Guide

This directory contains design system documentation and reference materials for maintaining visual consistency across the ML Learning Path application.

## Directory Structure

```
theme-guide/
├── DESIGN_SYSTEM.md          # Complete design system documentation
├── COMPARISON_GUIDE.md        # Side-by-side comparison with Compass
├── README.md                  # This file
├── compass-reference/         # Original Compass template files
│   ├── globals.css           # Compass CSS configuration
│   ├── typography.css        # Compass typography styles
│   ├── button-reference.tsx  # Compass Button component
│   └── sidebar-layout-reference.tsx  # Compass layout example
├── examples/                  # Code examples and patterns
└── assets/                    # Design assets (colors, screenshots)
```

## Quick Start

### For Developers

1. **Read the design system:** Start with `DESIGN_SYSTEM.md`
2. **Check component patterns:** Reference the Components section
3. **Compare implementation:** Use `COMPARISON_GUIDE.md` to verify alignment
4. **Reference Compass:** Look at `compass-reference/` for original patterns

### For Designers

1. **Color palette:** See Color System section in DESIGN_SYSTEM.md
2. **Typography scale:** Reference Typography section
3. **Component specs:** Components section has detailed specifications
4. **Figma/Sketch:** Use documented values for design tool setup

## Key Documents

### DESIGN_SYSTEM.md
**Complete design system specification including:**
- Color system with exact hex values
- Typography scale and font specifications
- Spacing and layout patterns
- Component library with code examples
- Accessibility guidelines
- Dark mode implementation

### COMPARISON_GUIDE.md
**Side-by-side comparison of:**
- Compass template vs our implementation
- What we kept, what we changed, why
- Visual verification checklist

### compass-reference/
**Original Compass template files for reference:**
- Copy files directly when needed
- Compare our implementation
- Verify we're following patterns correctly

## Design Principles

1. **Stay True to Compass:** Our design is based on the Compass template
2. **Consistency First:** Follow established patterns
3. **Accessibility Always:** WCAG 2.1 AA minimum
4. **Content First:** Design supports learning

## Usage Guidelines

### When Creating New Components

1. Check if similar component exists in DESIGN_SYSTEM.md
2. Review Compass reference implementation
3. Follow color palette exactly (no custom colors)
4. Use typography scale (no custom font sizes)
5. Apply dark mode styles
6. Test accessibility (keyboard, screen reader)

### When Modifying Existing Components

1. Verify changes align with design system
2. Update DESIGN_SYSTEM.md if adding patterns
3. Test in both light and dark mode
4. Verify accessibility isn't broken
5. Document why changes were needed

### When In Doubt

1. **Check Compass reference:** `compass-reference/` directory
2. **Read design system:** DESIGN_SYSTEM.md has answers
3. **Compare:** Use COMPARISON_GUIDE.md
4. **Ask questions:** Better to clarify than guess

## Color Palette Quick Reference

```
Gray Scale (Primary):
- gray-50:  #f9fafb (lightest)
- gray-100: #f3f4f6
- gray-400: #9ca3af (muted text dark mode)
- gray-700: #374151 (body text light mode)
- gray-950: #030712 (darkest, dark mode bg)

Accents:
- blue-500: #3b82f6 (focus states)
- red-600:  #dc2626 (destructive actions)
```

## Typography Quick Reference

```
Fonts:
- Sans: Inter Variable (body & UI)
- Mono: Geist Mono (code)

Scale:
- text-sm:   14px (default prose)
- text-base: 16px (large body)
- text-lg:   18px (h3)
- text-xl:   20px (h2)
- text-3xl:  32px (h1)
```

## Common Patterns

### Button
```tsx
<button className="rounded-full bg-gray-950 px-3.5 py-2 text-sm/6 font-semibold text-white hover:bg-gray-800">
  Click Me
</button>
```

### Card
```tsx
<div className="rounded-lg border border-gray-950/10 bg-white p-6 dark:border-white/10 dark:bg-gray-900">
  Content
</div>
```

### Prose Content
```tsx
<article className="prose max-w-none">
  {content}
</article>
```

## Maintenance

### Keeping In Sync with Compass

If Compass template updates:
1. Download new version
2. Update `compass-reference/` files
3. Review changes in COMPARISON_GUIDE.md
4. Update our implementation if needed
5. Update DESIGN_SYSTEM.md documentation

### Updating Documentation

When adding new patterns:
1. Document in DESIGN_SYSTEM.md
2. Add code example
3. Explain when/how to use
4. Include accessibility notes
5. Add dark mode variant

## Resources

- **Compass Template:** https://tailwindcss.com/templates/compass
- **Tailwind v4 Docs:** https://tailwindcss.com/docs
- **Inter Font:** https://rsms.me/inter/
- **WCAG Guidelines:** https://www.w3.org/WAI/WCAG21/quickref/

## Questions?

1. Check DESIGN_SYSTEM.md first
2. Look at compass-reference/ for original patterns
3. Review COMPARISON_GUIDE.md for implementation details
4. Still stuck? Ask the team!

---

**Remember:** This is a learning platform. Design should support learning, not distract from it. Keep it clean, simple, and accessible.
