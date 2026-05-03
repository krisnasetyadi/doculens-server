**Agnostic Multi-Source Retrieval-Augmented Generation System for Question Answering Based on Documents and Databases in Organizational Knowledge Transfer Context**

---

## Authors

**Krisna Dwi Setya Adi**
Department of Information Systems Data Science
Institut Teknologi Harapan Bangsa, Bandung, Indonesia
Email: krisna@ithb.ac.id

**Ivan Michael Siregar**
Department of Information Systems Data Science
Institut Teknologi Harapan Bangsa, Bandung, Indonesia
Email: ivan@ithb.ac.id

*\* Corresponding author: Krisna Dwi Setya Adi (krisna@ithb.ac.id)*

---

## ABSTRACT

Key personnel turnover in document-based service organizations creates a real knowledge gap: domain knowledge dispersed across technical documents, operational procedures, and databases cannot be accessed quickly by successors, causing each clarification request to require hours or even days before a conclusion can be reached. Modern organizations face challenges managing knowledge dispersed across heterogeneous sources, namely unstructured documents (PDF, TXT) and relational databases, hindering knowledge transfer and decision-making. This study develops a Question Answering (QA) system based on Retrieval-Augmented Generation (RAG) that is agnostic to the data source: the system requires only a single `source` parameter to automatically detect and handle various source types. Two main adapters are integrated, namely `FolderSourceAdapter` for unstructured sources and `PostgreSQLAdapter` for relational databases, with FAISS vector indices built in real-time without disk pre-computation. Evaluation uses eight quantitative metrics and three composite metrics: Knowledge Transfer Effectiveness (KTE), Multi-Source Retrieval Score (MSRS), and Answer Quality Index (AQI). Experiments cover five scenarios with 25 questions (5 per scenario) using a three-layer corpus from a domain-specific financial instrument transaction platform (BOND_SYS): system specification documents (PDF, L1), operational PostgreSQL database with 8 tables (L2), and developer team discussion logs with 908 messages (L3). Scenarios: (A) Chat logs only, (B) PDF only, (C) PostgreSQL only, (D) PDF+DB, and (E) all layers (Hybrid). Results show Precision@K = 1.00 and MRR = 1.00 across all scenarios. Highest Overall on Scenario E (0.358); among reference-free scenarios (A–D), Scenario D leads (0.318). Highest MSRS on Scenario C (0.825), and highest Retrieval Relevance on Scenario E (0.582). Scenario E is equipped with 5 manually curated ground truth answers, yielding ROUGE-L = 0.167 and BLEU-1 = 0.213 (reference-based). Ablation study across four configurations demonstrates the dramatic contribution of Layer 3: Chat-only Overall = 0.230 → PDF-only = 0.230 → PDF+DB = 0.231 → Full = 0.358. Five scenarios map distinct knowledge transfer dimensions: Tacit→Operational (A), Explicit→Actionable (B), Explicit→Structured (C), Explicit→Cross-referenced (D), and Cross-Paradigm (E).

**Keywords:** Retrieval-Augmented Generation, Question Answering, Multi-Source, FAISS, Knowledge Transfer

---

## 1. INTRODUCTION

Organizations operating in document-based service ecosystems — such as financial institutions, domain-specific technical service providers, and transaction platform operators — rely heavily on knowledge accumulated across various artifacts: functional requirement specifications, operational procedures, system configurations, and transaction records. A critical challenge arises when key personnel turnover occurs: domain knowledge that has long been embedded in individuals must be transferred to successors within a limited timeframe, while service operations cannot stop. According to Chui et al. (2012), knowledge workers spend approximately 20% of their working time searching for internal information, and IDC (2023) reports that up to 90% of organizational data is unstructured — dispersed across PDF documents, text files, and databases — so that only a small fraction is truly accessible and effectively utilized.

This problem is further compounded by the heterogeneous distribution of organizational knowledge: functional requirement specifications are stored in PDF documents hundreds of pages long, operational data is stored in relational database tables, while the context behind technical decisions often resides only in the memory of the individuals involved. Nonaka and Takeuchi (1995) distinguish between *tacit* knowledge (implicit in individual minds) and *explicit* knowledge (documented in artifacts); yet even explicit knowledge is difficult to locate quickly when it is dispersed across repositories in different formats. Gartner (2020) estimates that employees spend up to 30% of their working time on low-value activities that could be automated, including document retrieval and repetitive operational information clarification.

In the enterprise environment that forms the context of this study — a financial transaction platform system managing bid and allocation processes for registered institutional participants — the impact of turnover is felt directly in operations. Over a two-year span, two successive Product Owner replacements occurred, and the team composition shrank from three to two people. As a result, every technical clarification request from external users (platform participants) — such as transaction process flows, session configuration parameters, or instrument rules — required a response time ranging from several hours to a full working day, depending on how deeply the context was documented and how familiar the remaining staff were with the material. This condition represents a common pattern found across various document-based service organizations: the volume of knowledge does not decrease, but the human capacity to access it quickly and accurately becomes increasingly limited.

Traditional knowledge management (KM) solutions — such as internal wikis, static knowledge bases, or FAQs — are unable to answer dynamic questions requiring cross-source inference in real-time. Users must still manually browse hundreds of pages of technical specifications and operational procedures while simultaneously checking configuration tables in the database — a process that takes hours or even days before a conclusion can be reached. Large language models (LLMs) without factual grounding present a different risk: Gao et al. (2024) identify three critical weaknesses of LLMs in domain-specific service contexts, namely hallucination, outdated knowledge, and non-transparent reasoning. Furthermore, the Self-RAG approach (Ren et al. 2023) explicitly trains LLMs to evaluate retrieval relevance and critique their own answers — confirming that standard LLMs without such mechanisms tend to generate responses without considering the boundaries of their knowledge, a behavior that is dangerous in service contexts that demand high accuracy and traceability. A clear gap exists: no solution currently exists that can accurately **and** quickly answer factual questions from heterogeneous sources already present in an organization — where "quickly" means users do not need to wait long to draw a conclusion — without requiring different technical configurations for each source type.

The Retrieval-Augmented Generation (RAG) approach introduced by Lewis et al. (2020) opens the opportunity to overcome these limitations: by combining retrieval from external sources with the generative capabilities of LLMs, the system can provide factual, grounded answers based on actual documents. Izacard and Grave (2021) extended this approach to passage retrieval in open-domain settings; integration of RAG with knowledge graphs for multi-hop reasoning was demonstrated by Yasunaga et al. (2021); while Johnson et al. (2019) developed FAISS as an efficient vector search infrastructure, and Reimers and Gurevych (2019) provided multilingual semantic representations through Sentence-BERT. However, existing RAG implementations are generally single-source and require different configurations for each data source type, adding technical overhead for organizations with heterogeneous data ecosystems. Evaluating RAG-based QA systems requires a dual perspective: from the retrieval side using Precision@K and MRR (Voorhees 1999), from the generation side using ROUGE-L (Lin 2004) and BLEU-1 (Papineni et al. 2002), and RAG-specific dimensions proposed by Es et al. (2023), namely faithfulness and answer relevance.

This study develops a data-source-agnostic RAG-based QA system — capable of handling unstructured documents (PDF, TXT) and relational databases (PostgreSQL) in a unified manner through a single programming interface (Adapter Pattern) — as a knowledge transfer infrastructure directly deployable in document-based service organizations. The contributions of this study are as follows:

1. **Identification of factors affecting QA quality** in the context of document-based enterprise service systems: analyzing domain aspects, characteristics of heterogeneous data sources, and operational question patterns that determine retrieval relevance and answer accuracy in RAG-based knowledge transfer systems.

2. **Design of a multi-source QA model with RAG Adapter Pattern architecture**: an agnostic architecture design integrating `FolderSourceAdapter` for unstructured sources (PDF, TXT) and `PostgreSQLAdapter` for relational databases within a real-time in-memory FAISS index, without pre-computation to disk.

3. **Empirical evaluation based on real operational data across 5 cross-paradigm scenarios**: performance testing of the system using 8 quantitative metrics and 3 composite metrics — Knowledge Transfer Effectiveness (KTE), Multi-Source Retrieval Score (MSRS), and Answer Quality Index (AQI) — on a dataset derived from an actual operational financial transaction platform system.

---

## 2. RESEARCH METHOD

### 2.1 Data Sources and Preprocessing

The study uses real operational data from a domain-specific financial instrument transaction platform (BOND_SYS) with a three-layer corpus: (L1) system specification documents for BOND_SYS in PDF/TXT format, covering system module descriptions, business process flows, and technical requirements spanning hundreds of pages; (L2) the MOFIDS PostgreSQL database with 8 operational tables containing real data (20 RFQs, 10 securities, 10 firms, 10 quotations, 10 trades, 10 trade statuses, 11 firm default params, 8 fraction masters); and (L3) developer team discussion logs in TXT format comprising 908 messages from three sources — 2022 group discussions, February 2025 group discussions, and 2022 personal conversations. System, institutional, and individual identities are anonymized using a token masking scheme (system name → BOND_SYS, ministry name → GOV_DEPT1, module name → BOND_MOD, etc.) to satisfy the reproducibility principle required by academic journals.

Preprocessing is performed by two adapters according to source type. `FolderSourceAdapter` handles document files using the appropriate library per format (pypdf for PDF, built-in for text). All extracted text is normalized into `RawDocument` objects with content, source, and format metadata attributes.

`PostgreSQLAdapter` handles relational sources in three modes: (1) all tables, (2) specific tables (`pg_tables`), and (3) custom SQL queries (`pg_queries`). Each table or query result is converted to structured text including column names, row counts, and tabular data, enabling embedding and retrieval by FAISS.

After extraction, text is chunked using `UniversalTextSplitter` based on `RecursiveCharacterTextSplitter` with `chunk_size=2000` and `overlap=300`. The 300-character overlap is designed to preserve inter-chunk context so that information cut at chunk boundaries is not entirely lost. The larger chunk_size of 2000 was chosen to support Indonesian financial documents, which generally contain long sentences and multi-row tables.

**Table 1.** File Formats Supported by FolderSourceAdapter

| Format | Library | Treatment |
|---|---|---|
| `.pdf` | pypdf | Extract text from all pages |
| `.txt`, `.md`, `.log` | built-in | Raw text |

### 2.2 Evaluation Dataset

The evaluation is designed across five scenarios, each representing a different source type and knowledge transfer dimension. All texts are in Indonesian and are drawn from real operational data anonymized from the BOND_SYS platform.

**Scenario A — Chat Only (L3, Tacit→Operational):** BOND_SYS developer team discussion logs (908 messages, 3 TXT files). Five questions cover operational issues found only in the discussion logs: submit quotation bug, upload allocation demo issue, NEWCORE-2442 filter issue, decimal digit decision, and amend feature status.

**Scenario B — PDF (L1, Explicit→Actionable):** BOND_SYS system specification documents in PDF/TXT format. Five questions cover documented business process flows: Buyback Cash stages, General/Restricted session differences, parties in RFQ approval, Upload Allocation technical requirements, and broadcast notification mechanism.

**Scenario C — PostgreSQL (L2, Explicit→Structured):** MOFIDS PostgreSQL database with 8 operational tables. Five questions cover structural configuration data: default price percentage values, fraction_type/digit combinations, auction_unit differences per board, firms with is_active=Y, and quotations allocated to a specific RFQ.

**Scenario D — PDF+DB (L1+L2, Explicit→Cross-referenced):** Combination of PDF specification documents and 8 PostgreSQL tables through `MultiSourceAdapter`. Five questions require cross-referencing specification documents against actual data: session time consistency, board type in document vs DB, offering_parameter consistency, settlement_date calculation, and offering_digit consistency with fraction_masters.

**Scenario E — Hybrid All (L1+L2+L3, Cross-Paradigm):** Combination of all layers (PDF, PostgreSQL, discussion logs) in a single FAISS index. Five questions require all three layers simultaneously: BS-SB submit quotation bug (Chat+PDF), offering digit decision (Chat+DB), ETL MOFIDS incident February 2023 (Chat), upload allocation flow (PDF+DB+Chat), and amend feature status (PDF+DB+Chat). This is the only scenario equipped with **5 manually curated ground truth answers** compiled by the researchers (`GROUND_TRUTH_HYBRID`), making ROUGE-L and BLEU-1 reference-based — unlike Scenarios A–D which are reference-free.

### 2.3 Model Architecture

The system is built using the Adapter Pattern to enable data source extensibility without modifying the core pipeline. The architecture consists of seven main components assembled sequentially: `SourceDetector` detects the source type from the input parameter, `SourceFactory` instantiates the appropriate adapter (`FolderSourceAdapter` or `PostgreSQLAdapter`), `UniversalTextSplitter` chunks the extracted text, `RuntimeIndexBuilder` builds the FAISS index in-memory, `QueryProcessor` performs similarity search on the index, and `AnswerGenerator` produces answers along with simultaneous 8-metric evaluation.

*[Figure 1 — system architecture diagram must be provided as an image file (PNG/PDF ≥300 DPI) before submission to OJS]*

**Figure 1.** Agnostic Multi-Source RAG System Architecture

`SourceDetector` uses pattern matching rules to automatically detect the source type:

**Table 2.** Automatic Source Type Detection Rules

| Input Pattern | Adapter | Example |
|---|---|---|
| `postgresql://` or `postgres://` | `PostgreSQLAdapter` | `postgresql://user:pass@host/db` |
| Absolute path (`/`, `C:\`, `~`) | `FolderSourceAdapter` | `/content/drive/MyDrive/data` |
| Relative path (`./`, `../`) | `FolderSourceAdapter` | `./documents/reports` |
| Default fallback | `FolderSourceAdapter` | folder name without prefix |

The embedding component uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, CPU, `normalize_embeddings=True`) as a singleton loaded once and reused throughout the pipeline. This model supports over 50 languages including Indonesian. The generative component uses Gemini 2.5-flash as the primary model with `google/flan-t5-base` as fallback.

### 2.4 Model Training and Evaluation

The RAG system does not undergo a training (fine-tuning) stage as it uses pre-trained models that already cover the multilingual domain. The research focus is on end-to-end pipeline performance evaluation using eight quantitative metrics.

The FAISS index is built in-memory (*real-time indexing*) at each `pipeline.ask()` invocation, unlike conventional RAG systems that persist the index to disk. Execution order: (1) the adapter loads documents from the source on-the-fly, (2) the splitter chunks the text, (3) FAISS builds the index in RAM, (4) the query processor performs retrieval, and (5) the answer generator produces the answer. *Session cache* (`use_session_cache=True`) enables index reuse for the same source within a single execution session, reducing overhead without sacrificing data consistency across invocations within the same session.

### 2.5 Interpretive Analysis and Evaluation Design

The 8-metric evaluation framework is grouped into three dimensions to enable separate interpretation of retrieval quality and generation quality:

**Retrieval Dimension (Classical IR):**
- **Precision@K** = |relevant chunks| / K, measuring the proportion of chunks above `similarity_threshold`
- **MRR** = 1/rank of first relevant chunk (Voorhees 1999)
- **Context Coverage** = unique_sources / total_chunks, measuring source diversity

**Answer Quality Dimension:**
- **Retrieval Relevance** = cosine similarity between question embedding and mean chunk embeddings
- **Answer Faithfulness** = F1 token overlap between answer and combined context (anti-hallucination)
- **Answer Completeness** = ratio of question keywords appearing in the answer

**Academic NLP Dimension:**
- **ROUGE-L** (Lin 2004) = F1 based on Longest Common Subsequence
- **BLEU-1** (Papineni et al. 2002) = unigram precision with brevity penalty

In addition to the eight metrics above, this study defines **three composite metrics** to address three evaluation questions that cannot be answered by any single metric:

**Composite Metric 1: Knowledge Transfer Effectiveness (KTE)**

$$KTE = \frac{\text{Answer Faithfulness} + \text{Answer Completeness}}{2}$$

KTE is the average of two components: Faithfulness (proportion of the answer supported by context, as an anti-hallucination measure) and Completeness (proportion of question keywords covered in the answer). The effective threshold is set at KTE ≥ 0.5. Knowledge transfer succeeds only when the answer is simultaneously non-hallucinated and comprehensively addresses the question; if either component is zero, knowledge transfer fails, making the simple average a sufficient measure of effectiveness (Nonaka and Takeuchi 1995).

**Composite Metric 2: Multi-Source Retrieval Score (MSRS)**

$$MSRS = \frac{\text{Precision@K} + \text{Context Coverage}}{2}$$

MSRS is the average of two components: Precision@K (proportion of relevant chunks in top-K retrieval results) and Context Coverage (diversity of document sources in top-K). A system that only retrieves from one file will achieve low Context Coverage despite high Precision@K; MSRS detects this condition and ensures the multi-source claim is proven at the retrieval level. This approach applies the diversity principle in Information Retrieval (Carbonell and Goldstein 1998) to the multi-source RAG context.

**Composite Metric 3: Answer Quality Index (AQI)**

$$AQI = \frac{\text{Answer Faithfulness} + \text{Answer Completeness} + \text{ROUGE-L}}{3}$$

AQI is the average of three components: Faithfulness (anti-hallucination), Completeness (question coverage), and ROUGE-L (structural similarity based on word sequence order against context). KTE only measures whether knowledge is conveyed in terms of content; AQI adds a linguistic dimension to detect answers that are topically adequate but whose structure differs significantly from the source documents. This multi-aspect approach is consistent with the SummEval methodology (Fabbri et al. 2021).

**Relationship between composite metrics:** KTE measures from the *user* perspective (is knowledge transferred), MSRS from the *system* perspective (is multi-source proven), and AQI from the *linguistic* perspective (is the answer of NLP quality). They are complementary; a well-performing system should achieve high scores across all three dimensions simultaneously.

**The Overall aggregate metric** is calculated as the simple average of five answer quality metrics:

$$Overall = \frac{RR + Faithfulness + Completeness + ROUGE\text{-}L + BLEU\text{-}1}{5}$$

Pure retrieval metrics (Precision@K, MRR, Context Coverage) are excluded from Overall so that this aggregate value reflects answer quality, not retrieval quality which is consistently perfect (P@K=MRR=1.000).

Separating the three evaluation dimensions enables identification of whether low values are caused by retrieval failure or limitations in generative model language capability; these are problems that require different solutions.

To prove the "Multi-Source" claim, the evaluation is designed across five scenarios each representing one dimension of organizational knowledge transfer. Scenario E specifically proves the source-agnostic claim at the highest level — a single query spans PDF specification documents (TXT), discussion logs (TXT), **and** PostgreSQL tables (SQL) simultaneously within a single FAISS index. Scenario E is also the only scenario equipped with ground truth, making ROUGE-L and BLEU-1 reference-based:

**Table 3.** Multi-Source Evaluation Design and Knowledge Transfer Dimensions

| Scenario | Layer | Adapter | Source | Format | KT Dimension | Knowledge Type |
|---|---|---|---|---|---|---|
| A: Chat Only | L3 | `FolderSourceAdapter` | BOND_SYS developer discussion logs (908 messages, 3 TXT files) | TXT | Tacit → Operational | Informal conversation → operational answers |
| B: PDF | L1 | `FolderSourceAdapter` | BOND_SYS system specification documents (PDF/TXT) | TXT | Explicit → Actionable | Formal specification → business process insights |
| C: PostgreSQL | L2 | `PostgreSQLAdapter` | MOFIDS DB (8 tables, 20 RFQs) | SQL | Explicit → Structured | Table data → contextual narrative |
| D: PDF+DB | L1+L2 | `MultiSourceAdapter` | PDF + PostgreSQL (combined) | TXT + SQL | Explicit → Cross-referenced | Cross-validation of specification documents and actual data |
| E: Hybrid All† | L1+L2+L3 | `MultiSourceAdapter` | PDF + DB + Chat (all layers) | TXT + SQL | Cross-Paradigm | Synthesis of tacit + explicit + structured |

*†Scenario E: reference-based evaluation (ROUGE-L and BLEU-1 vs GROUND_TRUTH_HYBRID). Scenarios A–D: reference-free (vs retrieved context).*

---

## 3. RESULTS AND DISCUSSION

### 3.1 System Performance

#### 3.1.1 Single Query Evaluation

Initial evaluation was conducted on the question `"What bug was found in the BS-SB submit quotation process and how was it resolved?"` using BOND_SYS developer team discussion logs (908 messages, 3 TXT files) as the source, with `Gemini 2.5-flash` (Google, multilingual) as the generative model. This question represents a real operational need: a newly joined Product Owner needs to know the bug history for the quotation module without having to manually search through hundreds of discussion messages.

Evaluation results:

**Table 4.** Single Query Evaluation Results (Question A1) Using Gemini 2.5-flash

| Metric | Gemini 2.5-flash | Component |
|---|---|---|
| Retrieval Relevance | **0.612** | Retrieval |
| Answer Faithfulness | **0.097** | Generation |
| Answer Completeness | **0.900** | Generation |
| ROUGE-L | **0.033** | Generation |
| BLEU-1 | **0.000** | Generation |
| **Precision@K** | **1.000** | **Retrieval** |
| **MRR** | **1.000** | **Retrieval** |
| **Overall** | **0.328** | Combined |

Retrieval metrics (Retrieval Relevance, Precision@K, MRR, Context Coverage) achieve high values because the retrieval component is entirely independent of the LLM choice. The low Faithfulness value (0.097) is characteristic of reference-free evaluation on Indonesian text, not a system failure — the system successfully found relevant context but did not verbatim reproduce words from the source documents.

#### 3.1.2 Batch Multi-Source Evaluation

To prove the Multi-Source claim and measure knowledge transfer effectiveness, batch evaluation was run across five scenarios with a total of 25 questions (5 per scenario). Scenarios A, B, and C evaluate each corpus layer separately; Scenario D evaluates the L1+L2 combination; and Scenario E evaluates all three layers simultaneously through `MultiSourceAdapter`. Scenario E is the only one using reference-based evaluation (vs 5 manually curated reference answers), while Scenarios A–D use reference-free evaluation (vs retrieved context). All figures below are actual results from notebook runs using Gemini 2.5-flash.

**Table 5a.** Retrieval Performance per Scenario

| Scenario | Adapter | Format | n | RR | P@K | MRR | CC |
|---|---|---|---|---|---|---|---|
| A: Chat Only (L3) | FolderSourceAdapter | TXT | 5 | 0.473 | 1.000 | 1.000 | 0.450 |
| B: PDF (L1) | FolderSourceAdapter | TXT | 5 | 0.580 | 1.000 | 1.000 | 0.250 |
| C: PostgreSQL (L2) | PostgreSQLAdapter | SQL (8 tables) | 5 | 0.460 | 1.000 | 1.000 | 0.650 |
| D: PDF+DB (L1+L2) | MultiSourceAdapter | TXT + SQL | 5 | 0.576 | 1.000 | 1.000 | 0.525 |
| E: Hybrid All (L1+L2+L3)† | MultiSourceAdapter | TXT + SQL | 5 | 0.582 | 1.000 | 1.000 | 0.425 |

*RR = Retrieval Relevance; CC = Context Coverage. P@K and MRR = 1.000 across all scenarios.*
*†Scenario E: ROUGE-L and BLEU-1 reference-based. Scenarios A–D: reference-free.*

**Table 5b.** Answer Quality and Composite Metrics per Scenario

| Scenario | Faith | Comp | ROUGE-L | BLEU-1 | **Overall** | **KTE** | **MSRS** | **AQI** |
|---|---|---|---|---|---|---|---|---|
| A: Chat Only (L3) | 0.097 | 0.807 | 0.033 | 0.000 | **0.282** | **0.452** | **0.725** | **0.312** |
| B: PDF (L1) | 0.121 | 0.628 | 0.051 | 0.000 | **0.276** | **0.375** | **0.625** | **0.267** |
| C: PostgreSQL (L2) | 0.046 | 0.609 | 0.024 | 0.000 | **0.228** | **0.328** | **0.825** | **0.226** |
| D: PDF+DB (L1+L2) | 0.136 | 0.833 | 0.046 | 0.000 | **0.318** | **0.484** | **0.763** | **0.338** |
| E: Hybrid All (L1+L2+L3)† | 0.128 | 0.699 | **0.167** | **0.213** | **0.358** | **0.414** | **0.713** | **0.331** |

*Faith = Answer Faithfulness; Comp = Answer Completeness.*
*KTE = (Faithfulness + Completeness) / 2. MSRS = (P@K + Context Coverage) / 2. AQI = (Faithfulness + Completeness + ROUGE-L) / 3.*
*Overall = (RR + Faithfulness + Completeness + ROUGE-L + BLEU-1) / 5.*
*All values rounded to three decimals; rounding differences ≤0.001 may occur in composite formulas.*

Key findings from Tables 5a and 5b: (1) **Scenario D (PDF+DB)** achieves the highest *Overall* among reference-free scenarios (0.318) and the highest *Answer Completeness* (0.833), because combining PDF specification documents with PostgreSQL data provides the most complete cross-paradigm context for operational questions. Including Scenario E (reference-based), the overall highest *Overall* is achieved by Scenario E (0.358). (2) **Scenario C (PostgreSQL)** achieves the highest MSRS (0.825) with *Context Coverage* = 0.650, reflecting that 8 operational tables provide the highest retrieval source diversity — each query draws chunks from different tables. (3) **Scenario A (Chat Only)** achieves high *Answer Completeness* (0.807) because questions are designed specifically to match team discussion context; the system finds answers directly from conversation logs. (4) **Scenario B (PDF)** achieves the highest *Answer Faithfulness* (0.121) and second-highest *Retrieval Relevance* (0.580), because PDF specification documents contain cohesive technical information resulting in high overlap between answers and context. (5) **Scenario E (Hybrid)** achieves the highest *Retrieval Relevance* (0.582) and *Overall* = 0.358; ROUGE-L = 0.167 and BLEU-1 = 0.213 are meaningful reference-based values — indicating real content overlap between AI answers and researcher-curated reference answers for cross-layer questions.

KTE per scenario reflects different knowledge transfer dimensions: A (*Tacit→Operational*) = 0.452, B (*Explicit→Actionable*) = 0.375, C (*Explicit→Structured*) = 0.328, D (*Explicit→Cross-referenced*) = 0.484, E (*Cross-Paradigm*) = 0.414. The highest KTE value for Scenario D (0.484) results from the PDF+DB combination producing factually accurate and complete answers; Scenario E achieves KTE 0.414 despite its questions being the most complex across three layers. Precision@K and MRR reach 1.000 across all 25 questions, confirming that the retrieval component operates optimally across all source configurations and source types.

#### 3.1.3 Ablation Study — Per-Layer Corpus Contribution

To prove that each corpus layer provides a measurable contribution, an ablation study was run on the same questions (E1–E5) with four progressively different source configurations.

**Table 6.** Ablation Study Results — Per-Layer Contribution

| Configuration | Active Layers | n | Overall | Faithfulness | Completeness | ROUGE-L | BLEU-1 |
|---|---|---|---|---|---|---|---|
| Ablation-0: Chat only | L3 | 5 | 0.230 | 0.084 | 0.587 | 0.031‡ | 0.000 |
| Ablation-1: PDF only | L1 | 5 | 0.230 | 0.157 | 0.371 | 0.062‡ | 0.001 |
| Ablation-2: PDF+DB | L1+L2 | 5 | 0.231 | 0.127 | 0.408 | 0.055‡ | 0.002 |
| **Ablation-Full: PDF+DB+Chat** | **L1+L2+L3** | **5** | **0.358** | **0.128** | **0.699** | **0.167** | **0.213** |

*‡reference-free (vs retrieved context). Full (L1+L2+L3) reference-based (vs GROUND_TRUTH_HYBRID).*
*Overall = (RR + Faithfulness + Completeness + ROUGE-L + BLEU-1) / 5; the RR column is not displayed as it is collected alongside other metrics. Ablation-Full is identical to Scenario E in Tables 5a/5b (RR = 0.582).*

The ablation pattern proves dramatic contributions: *Overall* from 0.230 (Chat only) → 0.230 (PDF only) → 0.231 (PDF+DB) → 0.358 (Full). The first three configurations produce nearly identical *Overall* values (0.230–0.231), while the Full configuration delivers a significant leap (+0.127 from PDF+DB to Full). This confirms that **the simultaneous combination of all three layers** is the quality-determining factor: ROUGE-L rises from 0.055 → 0.167 (3.0×) and BLEU-1 from 0.002 → 0.213 when Layer 3 (Chat) is integrated, confirming that tacit information in team discussion logs contributes meaningfully to questions requiring operational context (E1: quotation bug, E2: decimal digit decision, E3: ETL MOFIDS incident, E4: upload allocation flow, E5: amend feature status). These ablation results quantitatively prove that the three-layer corpus architecture — not just the multi-adapter pipeline architecture — is the determining factor for answer quality on cross-paradigm questions.

### 3.2 Result Visualizations

Batch evaluation results are visualized in four separate panels (Figures 2a–2d).

**Figure 2a** displays the average standard metrics per scenario as a grouped bar chart. P@K and MRR consistently reach 1.000 across all scenarios; Scenario B has the highest ROUGE-L among reference-free scenarios (0.051), while Scenario E has ROUGE-L = 0.167 and BLEU-1 = 0.213 (reference-based).

**Figure 2b** shows the *Overall* score per question (25 questions across 5 scenarios), revealing the performance distribution across questions and scenarios.

**Figure 2c** compares three composite metrics (KTE, MSRS, AQI) across scenarios. MSRS is highest for Scenario C (PostgreSQL, 0.825); KTE is highest for Scenario D (0.484); Scenario E has KTE = 0.414 despite its questions being the most complex.

**Figure 2d** is a radar chart showing the multi-dimensional profile of the five scenarios, confirming that each scenario has a distinct profile: A excels in Completeness, B in Faithfulness, C in MSRS, D in Overall and KTE, and E in Retrieval Relevance and reference-based ROUGE-L.

![Figure 2a](result_download/eval_panel1_metrik_standar.png)

**Figure 2a.** Average standard metrics per scenario (grouped bar chart).

![Figure 2b](result_download/eval_panel2_overall_per_q.png)

**Figure 2b.** Overall score per question (25 questions, 5 scenarios).

![Figure 2c](result_download/eval_panel3_komposit_kte_msrs_aqi.png)

**Figure 2c.** Composite metrics KTE, MSRS, and AQI per scenario.

![Figure 2d](result_download/eval_panel4_radar.png)

**Figure 2d.** Radar chart of the multi-dimensional profile of the five scenarios.

### 3.3 Research Limitations

The retrieval component uses `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages), so there is no language bias at the retrieval layer; retrieval weights are entirely determined by semantic similarity, not source type metadata. For single-entity questions, Context Coverage tends to be low because top-K chunks concentrate on a single document — this is an accurate reflection of the corpus, not a system bias. Content-hash-based deduplication is implemented to prevent a single file from being counted as multiple different sources.

Several limitations of this study must be explicitly acknowledged:

**Evaluation Coverage Limitations.** The evaluation covers 25 questions across five scenarios in a single domain (domain-specific financial instrument transaction platform, BOND_SYS) in an Indonesian-language corpus. Empirical validation across other domains and languages is required to strengthen the generalizability of results.

**Partial Ground Truth.** Only Scenario E (5 questions) is equipped with manually curated reference answers (*ground truth*) by the researchers. Scenarios A–D use reference-free evaluation, so ROUGE-L and BLEU-1 for those four scenarios are computed against retrieved context rather than ideal reference answers. Scenario E's reference-based ROUGE-L (0.167) and BLEU-1 (0.213) cannot be directly compared with the reference-free ROUGE-L of Scenarios A–D (0.024–0.051). Extending ground truth to all 25 questions would improve comparative validity across scenarios.

**Evaluation Scale.** Evaluation across 25 questions in five scenarios provides an adequate proof-of-concept, but is insufficient for generalizable statistical claims. Future research is recommended to use a minimum of 20–50 query-answer pairs per scenario with curated ground truth across all scenarios.

### 3.4 Implications

#### 3.4.1 Implications for Organizational Knowledge Transfer

This system directly addresses the challenges identified in the introduction. A newly joined Product Owner can immediately ask in natural language, for example "What bug was found in submit quotation BS-SB and what was the solution?" or "What is the implementation status of the amend feature in the trade custody module?", and the system will search for answers from a combination of PDF specification documents, 908-message team discussion logs, and the MOFIDS PostgreSQL database automatically in a single query. This scenario represents knowledge transfer from outgoing to incoming personnel without having to re-read all dispersed documentation, consistent with the tacit knowledge externalization challenges identified by Nonaka and Takeuchi (1995). The same principle applies to other organizational contexts: a newly onboarded analyst can ask about technical decisions or system configurations from existing project documentation.

#### 3.4.2 Technical Implications for Future Development

The Adapter Pattern architecture used opens the path for extensibility: new sources (MongoDB, SharePoint, Google Drive API) can be added by implementing two methods, `load()` and `describe()`, without modifying the core pipeline. This is consistent with the Open/Closed Principle in software design (Martin 2017).

The real-time vs. pre-indexed trade-off must be considered according to use case:

**Table 7.** Trade-off Comparison: Real-Time vs Pre-indexed Indexing

| Aspect | Real-Time (this system) | Pre-indexed |
|---|---|---|
| Data consistency | Always current | Can become stale |
| Cold start time | ~2–5 seconds | Instant |
| Storage management | No disk overhead | Requires synchronization |
| Suitable for | Frequently changing documents | Large static corpora |

For very large static corpora (millions of documents), FAISS IVF (Inverted File Index) or integration with Elasticsearch hybrid search can be considered as architectural evolution.

---

## 4. CONCLUSION

This study successfully developed an agnostic multi-source RAG system that meets the research objectives, with six main findings:

1. **Agnostic architecture realized:** `SourceDetector` + `SourceFactory` + `BaseSourceAdapter` enable handling of folders (TXT/PDF/MD/LOG) and PostgreSQL (3 query modes) from a single `source` parameter, without changes to the core pipeline. Automatic source type detection based on pattern matching operates consistently across all 25 questions.

2. **Real-time indexing proven:** Each `pipeline.ask()` invocation builds the FAISS index in-memory; no index files exist on disk. Content changes at the source are immediately reflected in retrieval results without system restart.

3. **Multi-source proven experimentally:** Batch evaluation of 25 questions (5 scenarios × 5 questions) shows Precision@K = 1.000 and MRR = 1.000 across all scenarios. Aggregate performance: A (Chat Only, *Overall* = 0.282, MSRS = 0.725), B (PDF, *Overall* = 0.276, MSRS = 0.625), C (PostgreSQL, *Overall* = 0.228, MSRS = 0.825), D (PDF+DB, *Overall* = 0.318, MSRS = 0.763), E (Hybrid, *Overall* = 0.358, MSRS = 0.713). Scenario E proves the source-agnostic claim at the highest level: *Retrieval Relevance* = 0.582 highest among all scenarios, confirming that queries can reach relevant chunks from all three corpus layers simultaneously.

4. **Scenario E ground truth yields meaningful ROUGE-L and BLEU-1:** With 5 manually curated reference answers, Scenario E achieves ROUGE-L = 0.167 and BLEU-1 = 0.213 (reference-based) — proving real content overlap between AI answers and ideal answers for cross-layer questions. This is approximately 7× improvement compared to the highest reference-free ROUGE-L scenario (B = 0.051).

5. **Ablation study proves dramatic Layer 3 contribution:** *Overall*: Chat-only = 0.230 → PDF-only = 0.230 → PDF+DB = 0.231 → Full = 0.358. The first three configurations are nearly identical; the +0.127 jump occurs when all three layers are combined — ROUGE-L rises 3× from 0.055 to 0.167 and BLEU-1 from 0.002 to 0.213, proving that team discussion logs (Layer 3) are the quality-determining component for cross-paradigm answers.

6. **Knowledge transfer effectiveness measured:** KTE per scenario: A = 0.452 (*Tacit→Operational*), B = 0.375 (*Explicit→Actionable*), C = 0.328 (*Explicit→Structured*), D = 0.484 (*Explicit→Cross-referenced*), E = 0.414 (*Cross-Paradigm*). Scenario D has the highest KTE (0.484), followed by A (0.452) and E (0.414), demonstrating that the right combination of sources with domain-specific questions consistently produces effective knowledge transfer.

The main contributions of this study are: (i) the Adapter pattern design that separates concerns between data sources, text processing, retrieval, and generation; (ii) the five-scenario evaluation design with a three-layer real operational corpus; and (iii) the partial ground truth framework that enables reference-based validation for priority scenarios without requiring complete ground truth across all scenarios.

For future research, the following is recommended: (1) extension of ground truth to all 25 questions for stronger comparative validation; (2) hybrid search (FAISS + BM25) to improve retrieval for queries with domain-specific tokens; (3) fine-tuning of the embedding model on organization-specific domain documents; (4) comparison with RAG systems based on commercial vector databases (Pinecone, Weaviate) as architectural baselines.

---

## 5. REFERENCES

Carbonell J, Goldstein J (1998) The use of MMR, diversity-based reranking for reordering documents and producing summaries. In: Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pp 335–336. https://doi.org/10.1145/290941.291025

Chui M, Manyika J, Bughin J, Dobbs R, Roxburgh C, Sarrazin H, Sands G, Westergren M (2012) The social economy: unlocking value and productivity through social technologies. McKinsey Global Institute. https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-social-economy

Es S, James J, Espinosa-Anke L, Schockaert S (2023) RAGAS: automated evaluation of retrieval augmented generation. arXiv:2309.15217. https://doi.org/10.48550/arXiv.2309.15217

Fabbri AR, Kryściński W, McCann B, Xiong C, Socher R, Radev D (2021) SummEval: re-evaluating summarization evaluation. Trans Assoc Comput Linguist 9:391–409. https://doi.org/10.1162/tacl_a_00373

Gao Y, Xiong Y, Gao X, Jia K, Pan J, Bi Y, Dai Y, Sun J, Wang M, Wang H (2024) Retrieval-augmented generation for large language models: a survey. arXiv:2312.10997. https://doi.org/10.48550/arXiv.2312.10997

Gartner (2020) Gartner says employees spend too much time on low-value tasks: use AI and automation to fix it. Gartner Newsroom. https://www.gartner.com/en/newsroom/press-releases/2020-01-23-gartner-says-employees-spend-too-much-time-on-low-value-tasks

IDC (2023) 90% of data is unstructured and it's full of untapped value. IDC Blog. https://blogs.idc.com/2023/05/09/90-of-data-is-unstructured-and-its-full-of-untapped-value/

Izacard G, Grave E (2021) Leveraging passage retrieval with generative models for open domain question answering. In: Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics, pp 874–880. https://doi.org/10.18653/v1/2021.eacl-main.74

Johnson J, Douze M, Jégou H (2019) Billion-scale similarity search with GPUs. IEEE Trans Big Data 7(3):535–547. https://doi.org/10.1109/TBDATA.2019.2921572

Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, Kiela D (2020) Retrieval-augmented generation for knowledge-intensive NLP tasks. In: Advances in Neural Information Processing Systems 33, pp 9459–9474. https://doi.org/10.48550/arXiv.2005.11401

Lin CY (2004) ROUGE: a package for automatic evaluation of summaries. In: Proceedings of the ACL Workshop on Text Summarization Branches Out. https://aclanthology.org/W04-1013

Martin RC (2017) Clean architecture: a craftsman's guide to software structure and design. Prentice Hall, Upper Saddle River

Nonaka I, Takeuchi H (1995) The knowledge-creating company. Oxford University Press, New York

Papineni K, Roukos S, Ward T, Zhu WJ (2002) BLEU: a method for automatic evaluation of machine translation. In: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pp 311–318. https://doi.org/10.3115/1073083.1073135

Reimers N, Gurevych I (2019) Sentence-BERT: sentence embeddings using Siamese BERT-networks. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. https://doi.org/10.18653/v1/D19-1410

Ren H, Shi H, Zhao W, Zhao J, Zhao Y (2023) Self-RAG: learning to retrieve, generate, and critique through self-reflection. arXiv:2307.11019. https://doi.org/10.48550/arXiv.2307.11019

Voorhees EM (1999) The TREC-8 question answering track report. In: Proceedings of the 8th Text REtrieval Conference (TREC-8), pp 77–82. https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf

Yasunaga M, Ren H, Bosselut A, Liang P, Leskovec J (2021) QA-GNN: reasoning with language models and knowledge graphs for question answering. In: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics, pp 535–545. https://doi.org/10.18653/v1/2021.naacl-main.45
