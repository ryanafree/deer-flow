## 2024-07-19 - Intl Object Instantiation in AST Traversal
**Learning:** Instantiating `Intl` objects (like `Intl.Segmenter`) is extremely expensive and can become a severe performance bottleneck when done repeatedly inside a tight loop, such as iterating over nodes in an AST during markdown parsing/rendering.
**Action:** Always hoist `Intl` object instantiations outside of loops and AST traversal callbacks. Reusing a single module-level or memoized instance of an `Intl` object offers massive performance gains (~8x faster in microbenchmarks) for pure stateless operations.
