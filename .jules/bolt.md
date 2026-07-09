## 2025-02-14 - Rehype AST Traversal Bottleneck
**Learning:** Instantiating `Intl.Segmenter` inside a AST traversal function (like a Rehype plugin visiting every text node) creates a significant performance bottleneck during Markdown rendering, especially when streaming changes. V8/SpiderMonkey has high overhead for initializing `Intl` objects.
**Action:** Always instantiate `Intl` formatters and segmenters outside of loops and hot paths, reusing a singleton instance.
