## 2024-05-24 - Intl.Segmenter Performance Bottleneck
**Learning:** `Intl.Segmenter` instantiation is significantly slow (~0.3ms per instance). Creating it inside text-processing loops (like AST node visitors) causes severe O(N) performance degradation, especially during markdown rendering of long chat logs.
**Action:** Always cache and reuse `Intl` formatter/segmenter instances at the module level when their options are static, rather than recreating them per function call.
