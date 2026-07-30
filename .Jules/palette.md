## 2025-02-12 - Icon-only buttons lacking ARIA labels
**Learning:** In a shadcn-based Next.js app, icon-only buttons created with `<Button size="icon">` or `<Button size="icon-sm">` frequently miss `aria-label` attributes, making them inaccessible to screen readers. Relying solely on tooltips isn't enough for accessibility.
**Action:** Always add an explicit `aria-label` or ensure standard accessible components pass `aria-label` down to the button element.

## 2024-07-30 - Form Accessibility: Explicit Labels
**Learning:** Found instances of forms using only `placeholder` attributes on `<Input>` elements without explicit, associated labels, causing accessibility issues for screen readers.
**Action:** Always wrap inputs in a container and provide an explicit `<label>` element with an `htmlFor` attribute that matches the input's `id`. This ensures forms are accessible without relying solely on placeholders.
