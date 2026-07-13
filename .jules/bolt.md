## 2024-04-18 - Missing Memoization in Message List

**Learning:** `getMessageGroups` and `getAssistantTurnUsageMessages` were running O(N) recalculations on every re-render in `message-list.tsx` due to unrelated state changes (like `turnStartTime`).
**Action:** Consistently memoize operations scaling with message array size using `useMemo` in core message list components.
