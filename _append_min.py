# TO_REMOVE
# This file is a one-off script / dev utility and is no longer needed.
# Safe to delete after confirming no active references.
# -------------------------------------------------------------------
import pathlib

DIAGRAM = """

---

## Minimalist Version (journal Figure 1 — reviewer-friendly)

```mermaid
flowchart LR

    Q([\"User Question\"])
    SRC([\"Source String\nfolder path  /  postgresql://  /  folder|pg://\"])

    SD{\"SourceDetector\nauto-classifies source type\nFOLDER / POSTGRES / HYBRID\"}

    FA[\"FolderAdapter\nPDF · TXT · MD · LOG\npypdf / read_text\"]
    PA[\"PostgreSQLAdapter\n8 BOND_SYS tables\nSELECT * LIMIT 1000\"]
    MA[\"MultiSourceAdapter\nFolder + PostgreSQL\ncomposite load\"]

    SP[\"Text Splitter\nRecursiveCharacterTextSplitter\nchunk 2000 / overlap 300\"]
    EM[\"Embedding Model\nMiniLM-L12-v2\n384-dim per chunk\"]
    FI[(\"FAISS Index\nin-memory\nsession cache\")]

    QP[\"Query Processor\nembed question\ntop-8 chunks · threshold 0.2\"]
    GEN[\"Answer Generator\nGemini 2.5-flash\nzero-shot · retry + fallback\"]

    EV[\"Evaluator\nRR · AF · AC · ROUGE-L · BLEU-1\nP@K · MRR · CC\"]

    OUT([\"RAGResult\nAnswer + 8 Scores + Timing\"])

    Q   --> SD
    SRC --> SD
    SD  -->|\"FOLDER\"|   FA
    SD  -->|\"POSTGRES\"| PA
    SD  -->|\"HYBRID\"|   MA
    FA  --> SP
    PA  --> SP
    MA  --> SP
    SP  --> EM
    EM  --> FI
    FI  --> QP
    Q   -->|\"embed\"| QP
    QP  --> GEN
    GEN --> EV
    EV  --> OUT

    style Q   fill:#DBEAFE,stroke:#3B82F6,color:#1E3A5F
    style SRC fill:#DBEAFE,stroke:#3B82F6,color:#1E3A5F
    style SD  fill:#D1FAE5,stroke:#059669,color:#064E3B
    style FA  fill:#FEF3C7,stroke:#D97706,color:#78350F
    style PA  fill:#FEF3C7,stroke:#D97706,color:#78350F
    style MA  fill:#FDE8D8,stroke:#EA580C,color:#7C2D12
    style SP  fill:#EDE9FE,stroke:#7C3AED,color:#3B0764
    style EM  fill:#EDE9FE,stroke:#7C3AED,color:#3B0764
    style FI  fill:#EDE9FE,stroke:#7C3AED,color:#3B0764
    style QP  fill:#FCE7F3,stroke:#DB2777,color:#831843
    style GEN fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D
    style EV  fill:#E0F2FE,stroke:#0284C7,color:#0C4A6E
    style OUT fill:#D1FAE5,stroke:#059669,color:#064E3B
```

**Figure 1.** End-to-end architecture of the Agnostic Multi-Source RAG System.
`SourceDetector` classifies the input source string (FOLDER / POSTGRES / HYBRID) and
routes to the matching adapter. Documents are chunked (2,000 chars, 300 overlap),
encoded into 384-dim vectors by MiniLM-L12-v2, and indexed in-memory via FAISS with
session caching. At query time `QueryProcessor` retrieves the top-8 chunks
(similarity threshold 0.2) as context for `AnswerGenerator` (Gemini 2.5-flash,
zero-shot, automatic retry + model-fallback chain). The `Evaluator` scores the
answer simultaneously on 8 metrics: RR, AF, AC, ROUGE-L, BLEU-1, P@K, MRR, CC.
"""

p = pathlib.Path(r'c:\Users\RentalWorks-D5CKVT2\Documents\PDFREADER\pdf-reader\GAMBAR_1_ARSITEKTUR.md')
p.write_text(p.read_text(encoding='utf-8') + DIAGRAM, encoding='utf-8')
print('Done')
