## 2023-11-20 - [ARIA Label for Icon-Only ChatBox Artifact Close Button]
**Learning:** Icon-only buttons lacking accessible names are an issue for screen readers. Using the  hook correctly incorporates translations for  properties, such as `aria-label={t.common.close}` on `XIcon` buttons.
**Action:** Proactively identify  or  buttons that only contain an icon and no explicit  or , and ensure an  using internationalized text is present.
## 2023-11-20 - [ARIA Label for Icon-Only ChatBox Artifact Close Button]
**Learning:** Icon-only buttons lacking accessible names are an issue for screen readers. Using the useI18n hook correctly incorporates translations for aria-label properties, such as aria-label={t.common.close} on XIcon buttons.
**Action:** Proactively identify size="icon" or size="icon-sm" buttons that only contain an icon and no explicit aria-label or title, and ensure an aria-label using internationalized text is present.
