## 2024-07-11 - [Optimize array searches with backward iteration]
**Learning:** Found an anti-pattern in the React components where `.filter(condition)[array.length - 1]` was used to find the last item matching a condition. This forces a full $O(N)$ scan of the array and allocates $O(N)$ new memory.
**Action:** Replace this pattern with a backward `for` loop to search from the end. This stops exactly when the last match is found (avoiding unnecessary iterations) and requires $O(1)$ memory. Remember to add safety checks (like `if (step?.type)`) when iterating arrays directly.
