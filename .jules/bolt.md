## 2024-10-18 - Markdown Rendering Performance in Chat UI
**Learning:** In a chat interface with long context threads, parsing and rendering markdown (`rehype`, `remark`, etc.) for every single message during re-renders causes significant main-thread lag, even if the content itself hasn't changed.
**Action:** Always wrap heavy content rendering components like `MarkdownContent` in `React.memo` to skip unnecessary reconciliation passes during parent state updates (like new messages or typing animations). Ensure props like `rehypePlugins` passed down to it are memoized or stable.
