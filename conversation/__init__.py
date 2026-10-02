"""
conversation/
-------------
Everything that decides HOW the assistant replies before (or instead of)
the document-grounded RAG pipeline in processor.generate_hybrid_answer:

- language.py      — reply language (en/id) detected from the question itself,
                     so an English question never gets a forced-Indonesian answer.
- intent.py        — "conversation" (greeting, small talk, questions about
                     DocuLens features/plans) vs "retrieval" (the user is asking
                     about their own data). Only retrieval runs hybrid_search.
- guard.py         — refuses questions about how DocuLens itself is built
                     (tech stack, model, infra, prompts) and prompt-injection
                     attempts, before and after the LLM.
- feature_guide.py — the ONLY knowledge the conversation mode may answer from.
- messages.py      — canned bilingual replies (fallbacks, refusals, system
                     answers) that never go through the LLM.
"""
