## 2024-04-18 - Missing Memoization in Message List

**Learning:** `getMessageGroups` and `getAssistantTurnUsageMessages` were running O(N) recalculations on every re-render in `message-list.tsx` due to unrelated state changes (like `turnStartTime`).
**Action:** Consistently memoize operations scaling with message array size using `useMemo` in core message list components.

## 2024-07-11 - [Optimize array searches with backward iteration]
**Learning:** Found an anti-pattern in the React components where `.filter(condition)[array.length - 1]` was used to find the last item matching a condition. This forces a full $O(N)$ scan of the array and allocates $O(N)$ new memory.
**Action:** Replace this pattern with a backward `for` loop to search from the end. This stops exactly when the last match is found (avoiding unnecessary iterations) and requires $O(1)$ memory. Remember to add safety checks (like `if (step?.type)`) when iterating arrays directly.

## 2024-05-24 - Intl.Segmenter Performance Bottleneck
**Learning:** `Intl.Segmenter` instantiation is significantly slow (~0.3ms per instance). Creating it inside text-processing loops (like AST node visitors) causes severe O(N) performance degradation, especially during markdown rendering of long chat logs.
**Action:** Always cache and reuse `Intl` formatter/segmenter instances at the module level when their options are static, rather than recreating them per function call.

## 2024-10-18 - Markdown Rendering Performance in Chat UI
**Learning:** In a chat interface with long context threads, parsing and rendering markdown (`rehype`, `remark`, etc.) for every single message during re-renders causes significant main-thread lag, even if the content itself hasn't changed.
**Action:** Always wrap heavy content rendering components like `MarkdownContent` in `React.memo` to skip unnecessary reconciliation passes during parent state updates (like new messages or typing animations). Ensure props like `rehypePlugins` passed down to it are memoized or stable.
