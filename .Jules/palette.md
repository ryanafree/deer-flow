## 2023-10-25 - Missing ARIA Labels on Icon-only Buttons
**Learning:** The application uses several icon-only buttons via Shadcn UI's `size="icon"` or similar classes. These buttons often lack an `aria-label`, making them inaccessible to screen readers, which cannot interpret visual icons (like Trash or Copy).
**Action:** When using `size="icon"` (or similar styles where a button's content is visually only an icon), always explicitly add an `aria-label` attribute describing the button's action.
