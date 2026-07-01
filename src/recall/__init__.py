"""recall — machine-local hybrid knowledge recall + conversation curation.

A small, project-agnostic toolkit. The markdown knowledge corpus is the source
of truth; a disposable sqlite index (FTS5 keyword + sqlite-vec semantic, fused
with Reciprocal Rank Fusion) makes it searchable; a nightly curator distills
Claude Code conversations into the corpus. Every project recalls from its own
corpus plus a shared, machine-local global ("soul") corpus.

Generalized from a production predecessor into a standalone package.
"""
__version__ = "0.1.0"
