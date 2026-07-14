## 2025-02-12 - Icon-only buttons lacking ARIA labels
**Learning:** In a shadcn-based Next.js app, icon-only buttons created with `<Button size="icon">` or `<Button size="icon-sm">` frequently miss `aria-label` attributes, making them inaccessible to screen readers. Relying solely on tooltips isn't enough for accessibility.
**Action:** Always add an explicit `aria-label` or ensure standard accessible components pass `aria-label` down to the button element.
