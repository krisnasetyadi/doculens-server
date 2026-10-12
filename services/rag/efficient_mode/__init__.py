"""Efficient Mode — MS-247.

Opt-in, isolated context-compression experiment inspired by
https://github.com/juliusbrussee/caveman. Caveman itself is a Node
CLI/proxy (not a Python package) that sits between an agent and an LLM
provider; this backend never calls a provider directly on behalf of an
external agent, so instead of vendoring caveman we apply a small
rule-based compression pass, in-process, at the one place the fully
assembled RAG context already exists (processor.generate_hybrid_answer),
gated behind a request flag that defaults to False.

When the flag is off, callers get byte-identical behavior to before this
package existed — see compressor.compress_context's no-op guarantee.
"""
