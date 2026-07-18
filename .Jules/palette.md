## 2024-05-18 - Missing ARIA labels in Shadcn UI components
**Learning:** Icon-only buttons built with `Button` component and `size="icon"` from shadcn/ui do not inherently enforce accessible names. We must manually supply `aria-label` or visually hidden text for screen readers.
**Action:** When auditing the app, periodically search for `<Button .*size="icon"` and ensure each has a descriptive `aria-label` or `title` alongside it.
