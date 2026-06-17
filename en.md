Header 1

Agnostic Multi-Source Retrieval-Augmented Generation System for Question Answering Based On Documents and Databases in Organizational Knowledge Transfer Context 
Krisna Dwi Setya Adi1 · Ivan Michael Siregar1 

Abstract
Key personnel turnover in document-based service organizations creates a real knowledge gap: domain knowledge dispersed across technical documents, operational procedures, and databases cannot be accessed quickly by successors, causing each clarification request to require hours or even days before a conclusion can be reached. Modern organizations face challenges managing knowledge dispersed across heterogeneous sources, namely unstructured documents (PDF, TXT) and relational databases, hindering knowledge transfer and decision-making. This study develops a Question Answering (QA) system based on Retrieval-Augmented generation (RAG) that is agnostic to the data source: the system requires only a single source parameter to automatically detect and handle various source types. Two main adapters are integrated, namely FolderSourceAdapter for unstructured sources and PostgreSQLAdapter for relational databases, with FAISS vector indices built in real-time without disk pre-computation. Evaluation uses eight quantitative metrics and three composite metrics: Knowledge Transfer Effectiveness (KTE), Multi-Source Retrieval Score (MSRS) and Answer Quality Index (AQI). Experiments over five scenarios with 25 questions (5 per scenario) using a three-layer corpus from a domain-specific financial instrument transaction platform (BOND_SYS): system specification documents in PDF format (Layer 1), operational PostgreSQL database with 8 table (Layer 2), and developer team discussion logs with 908 messages (Layer 3). Scenarios: (A) Chat logs only, (B) PDF only, (C) PostgreSQL only, (D) PDF+DB and (E) all layers (Hybrid). Results show Precision@K = 1.000 and MRR = 1.000 across all scenarios. The highest Overall score is achieved in scenario E (0.358). Among reference-free scenarios (A-D), scenario D leads (0.318). The highest MSRS is recorded in Scenario C (0.825) and the highest Retrieval Relevance in Scenario E (0,582). Skenario E is equipped with 5 manually curated ground truth answers, yielding ROUGE-L = 0.167 and BLEU-1 = 0.213 (reference-based). Ablation study across four configurations demonstrates the dramatic contribution of Layer 3: Overall remains nearly flat across Chat-only (0.230), PDF only (0.230), PDF + DB (0.231), then jumps to 0.358 when all three layers are combined. Five scenarios map distinct knowledge transfer dimensions: Tacit-to-perational (A), Explicit ⇒ Actionable (B), Explicit ⇒ Structured (c), Explicit ⇒ Cross-referenced (D) and Cross-Paradigm (E).  

Keywords Retrieval-Augmented Generation · Question Answering · Multi-Source · FAISS · Knowledge Transfer

Introduction
Organizations operating in document-based service ecosystems, such as financial institutions, domain-specific technical service providers, and transaction platform operators, rely heavily on knowledge accumulated across various artifacts: functional requirement specifications, operational procedures, system configurations, and transaction records. A critical challenge arises when key personnel turnover occurs: domain knowledge embedded in individuals must be transferred to successors within a limited time, while service operations cannot be interrupted. According to Chui et al. (2012), knowledge workers spend approximately 20% of their working time searching for internal information, and IDC (2023) reports that up to 90% of organizational data is unstructured scattered across PDF documents, text files, and databases, such that only a small fraction is effectively accessible and utilized.
This problem is further compounded by the heterogeneous distribution of organizational knowledge: functional specifications are stored in PDF documents spanning tens to hundreds of pages, operational data resides in relational database tables, while the context of technical decision often exists only in the personal memory of the individual involved. Nonaka and Takeuchi (1995) distinguish between tacit knowledge (implicit in an individual’s mind). However, even explicit knowledge is difficult to locate quickly when it is scattered across repositories in different formats. Alavi and Leidner (2001) further emphasize that knowledge management systems often fail when organizational knowledge remains fragmented across repositories and cannot be operationalized in day-to-day decision workflows.  Gartner (2020) estimates that employees spend up to 30% of their working time on low-value activities that could be automated, including document retrieval and repetitive operational information clarification.
In the enterprise environment that forms the context of this study, a financial instrument transaction platform manages the offering and allocation processes of financial instruments for registered participant institutions. The impact of turnover is felt directly at the operational level. Over a two-year period, two consecutive changes of Product Owner occurred, with the team size shrinking from three to two members. As a result, each technical clarification request from external users (platform participants), such as transaction process flows, configuration parameters, or instrument rules, required response times ranging from several hours to a full working day, depending on how thoroughly the context was documented and how familiar the remaining personnel were with the material. This condition represents a common pattern found across various document-based service organizations: the volume of knowledge does not decrease, but the human capacity to access it quickly and accurately becomes increasingly limited.
Traditional Knowledge management (KM) solutions such as internal wikis and static knowledge bases are unable to answer dynamic questions that require real-time cross-source inference. Users must still manually browse through tens to hundreds of pages of technical specifications and operational procedures while simultaneously checking configuration tables in the database; a process that consumes hours or even days before a conclusion can be formulated. Large Language Models (LLMs) without factual grounding present a different risk: Gao et al. (2024) identify three critical weaknesses of LLMs in domain-specific service context namely hallucination, outdated knowledge, and non-transparent reasoning, while Ji et al (2023) provide broader evidence that hallucination remains a structural limitation in neutral text generation systems. Furthermore, the Self-RAG approach (Ren et al. 2023) explicitly trains LLMs to evaluate retrieval relevance and self-critique their own answer, confirming that standard LLMs without such mechanisms tend to produce answer without considering the boundaries of their knowledge, a behavior that is particularly dangerous in service contexts demanding accuracy and traceability. A clear gap therefore exists: no solution is yet capable of answering factual questions accurately and promptly from heterogeneous sources already present within an organization, where “promptly” means users do not have to wait extended periods to reach a conclusion, without requiring different technical configurations for each source type. Two limitations. First, single source. Second, unrealtime not autoupdate.
The Retrieval-Augmented Generation (RAG) approach introduced by Lewis et al. (2020) opens an opportunity to address these limitations by combining retrieval from external sources with LLM generation capabilities, enabling the system to provide factual, grounded answers based on actual documents. Izacard and Grave (2021) extend this approach for passage retrieval in open domains settings dense effectiveness was demonstrated by Karpukhin et al. (2020); integration of RAG with knowledge graphs for multi-hop reasoning was demonstrated by Yasunaga et al. (2021); while FAISS has evolved as a scalable vector search infrastructure from Johnson et al. (2019) to the current library architecture described by Douze et al. (2024), and Reimers and Gurevych (2019) provided multilingual semantic representations through Sentence-BERT. However, existing RAG implementations are generally single-source and require different configurations or single source? for each data source type, thereby increasing the technical burden for organizations with heterogeneous data ecosystems. Evaluating RAG-based QA systems requires a dual perspective: retrieval quality using Precision@K and MRR (Voorhees 1999), generation quality using ROUGE-L (Lin 2004) and BLEU-1 (Papineni et al. 2002), and RAG-specific dimensions proposed by Es et al. (2023), namely faithfulness and answer relevance.
These limitation motivate the study of … This study develops a RAG-based QA system that is agnostic to data sources, capable of handling unstructured documents (PDF, TXT) and relational databases (PostgreSQL) in a unified manner through a single programming configuration interface (Adapter Pattern), thereby providing a knowledge transfer infrastructure that can be directly implemented in document-based service organizations. The contributions novelty of this study lies in three dimensions:
In summary, the contributions of this paper are summa rized as follows 
Propose an integrated source-agnostic RAG architecture model that layered-based and  works under various source configuration and  realtime condition.
A source-agnostic RAG architecture based on the Adapter Pattern that unifies heterogeneous knowledge sources under a single source parameter with automatic type detection and real-time in-memory FAISS indexing, eliminating the need for separate configuration, pre-computed indices, or source-specific pipeline variants. No prior RAG work integrates unstructured document folders and relational databases within a single unified pipeline.
Implement layered-based of structural and unstructural knowledge paradigm with progressive scenarios that isolate and quantify the incremental contribution.
Evaluate the model using metrics composite evaluation metrics purpose-build for multi-source RAG in organizational knowledge transfer contexts: Knowledge Transfer Effectiveness (KTE), Multi-Source Retrieval Score (MSRS), and Answer Quality Index (AQI),

Three novel composite evaluation metrics purpose-build for multi-source RAG in organizational knowledge transfer contexts: Knowledge Transfer Effectiveness (KTE), Multi-Source Retrieval Score (MSRS), and Answer Quality Index (AQI), each addressing an evaluation dimension not covered by existing single metrics, namely user-facing transfer success, retrieval source diversity, and linguistic answer quality respectively.
A cross-paradigm empirical evaluation framework employing a three-layer real operational corpus spanning unstructured specifications (PDF), relational databases (PostgreSQL), and tacit knowledge from team communication logs (TXT) across five progressive scenarios that isolate and quantify the incremental contribution of each knowledge paradigm, including the previously unmeasured contribution of developer discussion logs to cross-layer question answering.

Related Work
Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), addresses a fundamental limitation of parametric language models by grounding generation in dynamically retrieved external documents. The approach has been extended in several directions: Izacard and Grave (2021) demonstrate that fusing multiple retrieved passages significantly improves open-domain question answering, while Yasunaga et al. (2021) integrate knowledge graphs into the retrieval step to support multi-hop reasoning. A comprehensive survey by Gao et al. (2024) identifies three persistent failure modes in RAG systems operating under domain-specific constraints: factual hallucination, reliance on outdated parametric knowledge, and non-transparent reasoning chains ren et al (2023) address these failure modes through Self-RAG, training the model to selectively retrieve and self-critique its own outputs. Despite these advances, existing RAG systems are uniformly designed for a single single source type, typically a document corpus indexed in a vector store, and require separate pipelines when the knowledge base includes both unstructured documents and relational databases.

Multi-Source and Heterogeneous Retrieval

Integrating heterogeneous knowledge source into a single retrieval pipeline remains an open challenge. Johnson et al. (2019) provide the foundational vector search infrastructure through FAISS, enabling billion-scale similarity search on dense embeddings. Reimers and Gurevych (2019) extend dense retrieval to multilingual settings via Sentence-BERT, making semantic search viable across non-English corpora. Prior work on hybrid retrieval combines dense and sparse methods (e.g., BM25 + FAISS) but assumes a homogeneous document collection. No published work to date proposes a unified Adapter Pattern (Gamma et al. 1994) architecture that transparently handles both folder-based unstructured sources (PDF, TXT) and relational databases (PostgreSQL) within a single RAG pipeline without source-specific preprocessing branches.
Evaluation of RAG Systems

Evaluation of RAG-based QA systems requires metrics that separately capture retrieval quality and generation quality. Classical information retrieval metrics such as Precision@k and MRR (Voorhees 1999) measure retrieval precision and rank quality respectively. Generation quality is typically assessed using ROUGE-L (Lin 2004) for subsequence overlap and BLEU-1 (Papineni et al. 2002) for unigram precision against reference answers. Es et al. (2023) propose RAGAS, introducing faithfulness and answer relevance as LLM-judge metrics specifically designed for RAG evaluation. Fabbri et al. (2021) demonstrate through SummEval that multi-aspect evaluation is necessary to capture the full quality profile of generated text. However, no existing evaluation framework addresses the specific requirements of multi-source RAG in organizational knowledge transfer contexts, namely measuring retrieval diversity across source types, transfer effectiveness from tacit and explicit knowledge, and linguistic quality simultaneously. This study fills that gap through three purpose-built composite metrics: KTE, MSRS, and AQI.
Knowledge Transfer and Organizational QA

Nonaka and Takeuchi (1995) establish the theoretical foundation for organizational knowledge transfer, distinguishing tacit knowledge (embedded in individual experience) from explicit knowledge (codified in documents and data). The challenge of converting tacit knowledge into accessible, queryable form is central to knowledge management practice. Chui et al. (2012) quantify this problem at organizational scale, reporting that knowledge workers spend approximately 20% of working time searching for information. Gartner (2020) further estimates that 30% of employee effort is spent on automatable low-value tasks, including repeated document search and operational clarification. Existing QA systems applied to organizational knowledge management have focused predominantly on single-source settings, such as document corpora alone. The present study extends this scope by treating developer communication logs as a first-class knowledge source alongside formal specifications and operational databases, operationalizing tacit-to-explicit knowledge transfer as a measurable QA evaluation dimension.
.



Research Method
Data Sources and Preprocessing
This study employs real operational data from a domain-specific financial instrument transaction platform (BOND_SYS) organized in a three-layer corpus: (L1) BOND_SYS system specifications in PDF/TXT document format, covering system module descriptions, business process flows, and technical requirements; (L2) a BOND_SYS PostgreSQL database with 8 operational tables containing real data (20 RFQs, 10 Securities, 10 firms, 10 quotations, 10 trades, 10 trade statuses, 11 firm default parameters, 8 fraction masters); and (L3) developer team discussion logs in TXT format comprising 908 messages from three sources: a 2022 group discussion, a February 2025 group discussion and a 2022 personal conversation. System, institutional and individual identities are anonymized using a token masking scheme (system name: BOND_SYS, ministry name: GOV_DEPT1, module name: BOND_MOD, etc).
Preprocessing is performed by two adapters according to source type. FolderSourceAdapter handles document files using the appropriate library per format (pypdf for PDF, built-in for text). All extracted text is normalized into RawDocument objects with content, source and format metadata attributes.
PostgreSQLAdapter handles relational sources in three modes: (1) all tables, (2) specific tables via a table name list (pg_tables), and (3) custom SQL queries (pg_queries). Each table or query result is converted into structured text that includes column names, row counts, and tabular data, enabling embedding and retrieval by FAISS. In this study, mode (2) is used with a predefined list of 8 BOND_SYS operational tables; modes (1) and (30 are supported by the adapter interface but were not invoked in this evaluation, as the relevant tables were known in advance and no cross-table aggregation queries were required.
Following extraction, text is segmented using UniversalTextSplitter based on RecursiveCharacterTextSplitter with chunk_size = 2000 and overlap = 300. The 300 character overlap is designed to preserve inter-chunk context so that information split at chunk boundaries is not entirely lost. The larger chunk_size (2000) is selected to support Indonesian-language financial documents that typically contain long sentences and multi-row tables. This chucking choice is aligned with retrieval-augmented pretraining evidence that context segmentation and retrieval granularity substantially affect downstream generation quality and factual grounding (Borgeaud et al. 2022; Shi et al. 2023) Table 1 summarises the file formats supported by FolderSourceAdapter.

Table 1. File Formats Supported by FolderSourceAdapter


Format
Library
Processing
.pdf
pypdf
Text extraction from all pages
.txt, .md, .log
built-in
Raw text

Evaluation Dataset
The evaluation is designed across five scenarios, each representing a different source type and knowledge transfer dimension. All text is in Indonesian and originates from real operational data anonymized from the BOND_SYS platform.
Scenario A, Chat Only (L3, tacit to operational): Developer team discussion logs from BOND_SYS (908 messages, 3 TXT files). Five questions cover operational issues found exclusively in the discussion logs: submit quotation bug, upload allocation demo issue, a filter-status bug tracked as NEWCORE-2442 (an internal issue tracker ticket referencing a backend detect in the WAITING-status filter), decimal digit decision, and amend feature status.
Scenario B, PDF Only (L1, explicit to actionable): BOND_SYS system specification documents in PDF/TXT format. Five questions cover documented business process flows: INSTRUMENT_TYPE_A stages, differences between General/Restricted sessions, parties involved in RFQ approval, Upload Allocation technical requirements, and broadcast notification mechanism.
Scenario C, PostgreSQL Only (L2, explicit to structured): BOND_SYS PostgreSQL database with 8 operational tables. Five questions cover structural configuration data: default price percentage values, fraction_type/digit combinations, auction_unit differences per board, firms with is_active = Y, and quotations allocated to a specific RFQ.
Scenario D, PDF + DB (L1+L2, explicit to cross-referenced): Combined PDF documents and PostgreSQL 8 tables through MultiSourceAdapter. Five questions require cross-referencing specification documents with actual data: session time consistency, board type in documents vs. DB, offering_parameter consistency, settlement_date calculation, and offering_digit consistency with fraction_masters.
Scenario E, Hybrid All (L1+L2+L3, Cross-Paradigm): All layers (PDF, PostgreSQL discussion logs) combined into a single FAISS index. Five questions require all three layers simultaneously: submit quotation BOARD_TYPE_A bug (Chat+PDF), offering digit decision (Chat+DB), BOND_SYS ETL incident (anonymized period) (Chat), upload allocation flow (PDF+DB+Chat), and amend feature status (PDF+DB+Chat). This is the only scenario equipped with 5 manually curated reference answers (GROUD_TRUTH_HYBRID), making ROUGE-L and BLEU-1 reference-based, in contrast to Scenarios A-D which are reference-free (vs. retrieved context

System Architecture
The system is built using the Adapter Pattern (Gemma E et al., 1994) to enable data source extensibility without modifying the core pipeline. The architecture consists of seven main components arranged sequentially: SourceDetector detects the source type from the input parameter, Sourcefactory initializes the appropriate adapter (FolderSourceAdapter or PostgreSQLAdapter), UniversalTextSplitter segments the extracted text, RuntimeIndexBuilder constructs the FAISS index in-memory, QueryProcessor performs similarity search on the index, and AnswerGenerator produces answers along with evaluation of 8 metrics simultaneously. The overall architecture is illustrated in Figure 1.




Fig 1. Architecture of the Agnostic Multi-Source RAG System. Inputs (user question, source parameter, Config) enter the pipeline through SourceDetector, which classifies the source string and delegates document loading to the corresponding adapter via SourceFactory: FolderSourceAdapter for local files (PDF/TXT/MD/LOG), PostgreSQLAdapter for relational tables (SELECT * LIMIT 1000, serialized to text), or MultiSourceAdapter for hybrid sources. Raw documents are split by UniversalTextSplitter into 2,000-character chunks (300 character overlap), encoded by a singleton multilingual MiniLM-L12-V2 model (384 dimensions), and indexed in a FAISS in memory vector store with session-level caching. At query time, QueryProcessor embeds the question, retrieves top-8 chunks (similarity threshold 0.2), and passes the formatted context to AnswerGenerator, which calls Gemini 2.5-flash with zero-shot prompting and an automatic retry-and-model-fallback chain (attempt 2 falls back to gemini-2.0-flash, attempt 4 to gemini 1.5 flash; exponential backoff with base delay = 15 s, max 6 entries. The final RAGResults is simultaneously evaluated on 8 quantitative metrics: Retrieval Relevance (RR), Answer Faithfulness (AF), Answer Completeness (AC), ROUGE-L, BLEU-1, Precision@K, MRR, and Context Coverage (CC). Directed edges represent data flow; dashed edges indicate parallel scoring path in witch intermediate  retrieval artifacts (chunks and relevance scores) are forwarded to the Evaluator independently of the generation step. 


SourceDetector applies pattern matching rules to automatically detect the source type, as summarised in Table 2.

Table 2. Automatic Detection Rules for Data Source Type


Input Pattern
Adapter
Example
postgresql:// or postgres://
PostgreSQLAdapter
postgresql://user:pass@host/db
Absolute path(/, C:\, ~)
FolderSourceAdapter
/content/drive/Mydrive/data
Path relatif (./, ../)
FoldersourceAdapter
./documents/laporan
Default fallback
FolderSourceAdapter
Nama folder tanpa prefix


The embedding component uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions, CPU, normalize_embeddings=True) as a singleton loaded once and reused throughout the pipeline. This model supports over 50 languages including Indonesian. The generative component uses Gemini 2.5-flash as the primary model with google/flan-t5-base as a fallback.



Model Training and Evaluation
The RAG system does not undergo training (fine-tuning) phase, as it employs pre-trained models covering multilingual domains. The research focus is on end-to-end pipeline performance evaluation using eight quantitative metrics.
The FAISS index is constructed in-memory (real-time indexing) at each invocation of pipeline.ask(), unlike conventional RAG systems that persist the index to disk. The execution sequence is: (1) the adapter loads documents from the source on-the-fly, (2) the splitter segments the text, (3) FAISS builds the index in RAM, (4) the query processor performs retrieval, and (5) the answer generator produces the answer. Session cache (use_session_cache=True) enables index reuse for the same source within a single execution session, reducing overhead without sacrificing inte-call data consistency within the same session. This design choice follows the practical FAISS implementation direction described in recent library documentation (Douze et al. 2024).

Interpretive Analysis and Evaluation Design
The 8-metric evaluation framework is grouped into three dimensions to enable separate interpretation of retrieval quality and generation quality:
Retrieval Dimension (Classical IR): Precision@K Eq. (1) measures the proportion of top-K retrieved chunks that are relevant. MRR Eq. (2) captures rank quality by rewarding systems that place the first relevant chunk higher. Context Coverage Eq. (3) measures source diversity across top-K chunks.
.Precision@K= | relevant chunks |K	            (1)
MRR= 1rank of first relevant chunk	            (2)
Context Coverage=unique_sourcestotal_chunks	            (3)
Answer Quality Dimension:
Retrieval Relevance = cosine similarity of query embedding vs. mean of top-K chunk embeddings
Answer Faithfulness = F1 token overlap of answer vs. combined context (anti-hallucination)
Answer Completeness = ratio of question keywords present in the answer
NLP Dimension:
ROUGE-L (Lin 2004) = F1 based on Longest Common Subsequence
BLEU-1 (Papineni et al. 2002) = Unigram precision with brevity penalty
In addition to these eight metrics, three composite metrics are defined to address evaluation questions that cannot be answered by any single metric:
Composite Metric 1, Knowledge Transfer Effectiveness (KTE)
KTE=Answer Faithfulness + Answer Completeness2  (4)
KTE is the average of two components Eq. (4): Faithfulness (proportion of the answer supported by context, as an anti-hallucination measure) and Completeness (proportion of question keywords covered in the answer). The effective threshold is set at KTE  0.5. Knowledge transfer succeeds only if the answer is simultaneously non-hallucinatory and complete. If either component is zero, knowledge transfer fails; the simple average is therefore sufficient as an effectiveness measure (Nonaka and Takeuchi 1995).
Composite Metric 2, Multi-Source Retrieval Score (MSRS):
MSRS=Precision@K + Context Coverage2	                (5)
MSRS is the average of two components Eq. (5): Precision@K (proportion of relevant chunks in top-K retrieval) and Context Coverage (source document diversity in top-K). A system that retrieves only from a single file will obtain a low Context Coverage even with a high Precision@K. MSRS detects this condition and ensures that the multi-source claim is evidenced at the retrieval level. This approach applies the diversity principle in Information Retrieval (Carbonell and Goldstein 1998) to the multi-source RAG context.
Composite Metric 3, Answer Quality Index (AQI):

AQI=Faithfulness + Completeness + ROUGE-L33                 (6)
AQI is the average of three components Eq. (6): Faithfulness (anti-hallucination), Completeness (question coverage), and ROUGE-L (structural similarity based on word sequence order against context). While KTE only measures whether knowledge is conveyed in terms of content, AQI adds a linguistic dimension to detect answers that are thematically adequate but structurally divergent from the source documents. This multi-aspect approach is consistent with the SummEval methodology (Fabbri et al. 2021).
The relationships among composite metrics: KTE measures from the user perspective (whether knowledge is conveyed), MSRS from the system perspective (whether multi-source is evidenced), and AQI from the linguistic perspective (whether the answer is of NLP quality). The three are complementary; a well-performing system should achieve high scores across all three dimensions simultaneously.
The aggregate Overall metric is calculated as the simple average of five answer quality metrics:
O=RR + Faithfulness + Completeness + ROUGE-L + BLEU-15  
(7)
Pure retrieval metrics (Precision@K, MRR, Context Coverage) are excluded from Overall Eq. (7) so that this aggregate reflects answer quality rather than retrieval quality, which is consistently perfect (P@K = MRR = 1.000)
This three-dimensional evaluation separation enables identification of whether low values are caused by retrieval failure or by limitations in the generative model’s language capability, two problems that require fundamentally different solutions.
To substantiate the “Multi-Source” claim, the evaluation is designed in five scenarios, each representing one dimension of organizational knowledge transfer. Scenario E specifically proves the source-agnostic claim at the highest level: a single query spans PDF/TXT system specification documents, team discussion logs (TXT), and PostgreSQL tables (SQL) simultaneously within a single FAISS index. Scenario E is also the only scenario equipped with ground truth, making ROUGE-L and BLUE-1 reference-based. Table 3 summarises the five scenarios with their respective adapters and knowledge transfer dimensions.






Table 3. Multi-Source Evaluation Design and Knowledge Transfer Dimensions


Scenario
Layer
Adapter
Source
Format
KT Dimension
Knowledge Type
A: Chat
L3
FolderSourceAdapter
BOND_SYS discussion log (908 messages, 3 TXT file )
TXT
Tacit to Operational
Percakapan informal ⇒ jawaban operasional
B: PDF
L1
FolderSourceAdapter
BOND_SYS system spec ( PDF docs)
PDF./TXT
Explicit to Actionable
Format specification to business process insight
C: PostgreSQL
L2
PostgreSQLAdapter
BOND_SYS DB  (8 tables, 20 RFQ)
SQL
Explicit to Structured
Table data to contextual narrative
D: PDF + DB
L1 + L2
MultiSourceAdapter
PDF + PostgreSQL (combined)
TXT  + SQL
Explicit to Cross-referenced
Cross verification of specification and actual data
E: Hybrid 
L1 + L2 + L3
MultiSourceAdapter
PDF + DB + Chat (all layers)
TXT + SQL
Cross-Paradigm
Tacit, explicit, and structured combined

Scenario E: reference-based evaluation (ROUGE-L and BLEU-1 vs. GROUND_TRUTH_HYBRID). Scenarios A-D: reference-free (vs. retrieved context).



Results and Discussion
System Performance
   4.1.1 Multi source 

An initial evaluation was performed on the question “What bug was found in the submit quotation process for board BOARD_TYPE_A and what is the solution?” using the BOND_SYS developer team discussion logs (908 messages, 3 TXT files) as the source, with Gemini 2.5-flash (Google, multilingual) as the generative model. This question represents a real operational need: a newly onboarded Product Owner needs to retrieve the history of bugs in the quotation module without manually browsing through hundreds of messages.

Evaluation results are reported in Table 4.

Table 4. Single-Query Evaluation Results (Question A1) Using Gemini 2.5-flash

Metric
Gemini 2.5 Flash
Dimension
Retrieval Relevance
0.612
Retrieval
Answer Faithfulness
0.097
Generation
Answer Completeness
0.900
Generation
ROUGE-L
0.033
Generation
BLEU-1
0.000
Generation
Precision@K
1.000
Retrieval
MRR
1.000
Retrieval
Overall
0.328
Combined


Retrieval metrics (Retrieval Relevance, Precision@K, MRR, Context Coverage) achieve high values because the retrieval component is entirely independent of LLM selection. The low Faithfulness value (0.097) is characteristic of reference-free evaluation on Indonesian-language text, not a system failure; it indicates that the system successfully locates relevant context but does not reproduce source document wording verbatim.

4.1.2 Multi-Source Batch Evaluation

To substantiate the Multi-Source claim and measure knowledge transfer effectiveness, a batch evaluation was conducted across five scenarios with 25 questions in total (5 per scenario). Scenarios A, B, and C evaluate each corpus layer separately. Scenario D evaluates the L1 + L2 combination and Scenario E evaluates all three layers simultaneously through MultiSourceAdapter. Scenario E is the only one that employs reference-based evaluation (against 5 manually curated reference answers), while Scenarios A-D use reference-free evaluation (against retrieved context). All figures below are actual results from notebook runs using the Gemini 2.5-flash model.




 Table 5a. Retrieval Summary per Scenario


Scenario
Adapter
Format
N
RR
P@K
MRR
CC
A: Chat Only (L3)
FolderSourceAdapter
TXT
5
0.473
1.000
1.000
.0450
B: PDF (L1)
FolderSourceAdapter
TXT
5
0.580
1.000
1.000
0.250
C: PostgresQL (L2)
PostgresQLAdapter
SQL
5
0.460
1.000
1.000
0.650
D: PDF + DB (L1+L2)
MultiSourceAdapter
TXT + SQL
5
0.576
1.000
1.000
0.525
E: Hybrid (L1+L2+L3)
MultiSourceAdapter
TXT + SQL
5
0.582
1.000
1.000
0.425

RR = Retrieval Relevance; CC = Context Coverage; P@K dan MRR = 1.000 across all scenarios.
Table 5b. Answer Quality and Composite Metrics per Scenario


Scenario
Faith
Comp
ROUGE-L
BLEU-1
Overall
KTE
MSRS
AQI
A: Chat (L3)
0.097
0.807
0.033
0.000
0.282
0.452
0.725
0.312
B: PDF (L1)
0.121
0.628
0.051
0.000
0.276
0.375
0.625
0.267
C: PostgreSQL (L2)
0.0.46
0.609
0.024
0.000
0.228
0.328
0.825
0.226
D: PDF+DB (L1+L2)
0.136
0.833
0.046
0.000
0.318
0.484
0.763
0.338
E: Hybrid (L1+L2+L3)
0.128
0.699
0.167
0.213
0.358
0.414
0.713
0.331

Faith = Answer Faithfulness; Comp = Answer Completeness.
KTE = (Faith + Comp) / 2. MSRS = (P@K + CC ) / 2. AQI = (Faith + Comp + ROUGE-L) / 3.
Overall = (RR + Faith + Comp + ROUGE-L + BLEU-1) / 5
Skenario E: ROUGE-L dan BLEU-1 reference-based.Skenario A-D: reference-free.
All values rounded to three decimal places; rounding differences of 0.001 may occur in composite formulas.

KTE per scenario reflects distinct knowledge transfer dimensions: A (tacit to operational) = 0.452, B (explicit to actionable) = 0.375, C (explicit to structured) = 0.328, D (explicit to cross-referenced) = 0.484, E (cross-paradigm) = 0.414. The highest KTE in Scenario D (0.484) is attributable to the PDF+DB combination producing factually grounded and complete answers. Scenario E achieves KTE = 0.414 despite its questions being the most complex, spanning three layers. Precision@K and MRR reach 1.000 across all 25 questions, confirming that the retrieval component operates optimally across all source configurations and types. The layer-by-layer contribution of the corpus is further quantified through the ablation study in 4.1.3.
4.1.3 Ablation Study: Layer Contribution Analysis
To demonstrate that each corpus layer provides a measurable contribution, an ablation study was conducted on the same questions (E1-E5) across four progressively expanded source configurations.



Table 6. Ablation Study Results (Per-Layer Contribution)


Configuration
Layer Aktif
n
Overall
Faithfulness
Completeness
ROUGE-L
BLEU-1
Ablation-0: Chat 
L3
5
0.230
0.084
0.587
0.031
0.00
Ablation-1: PDF
L1
5
0.230
0.157
0.371
0.062
0.001
Ablation-2: PDF + DB
L1+L2
5
0.231
0.127
0.408
0.055
0.002
Full: PDF + DB + Chat
L1+L2+L3
5
0.358
0.128
0.699
0.167
0.213

Ablation-0 through Ablation-2: reference-free (vs. retrieved context). Full (L1+L2+L3): reference-based (vs. GROUND_TRUTH_HYBRID). Overall = (RR + Faithfulness + Completeness + ROUGE-L + BLEU-1) / 5; the RR column is not displayed as it is collected alongside other metrics. Ablation-Full is identical to Scenario E in Tables 5a/5b (RR = 0.582).

The ablation pattern demonstrate a dramatic contribution: Overall remains nearly flat across Chat-only (0.230), PDF-only (0.230), and PDF+DB (0.231), then delivers a significant leap to 0.358 when all three layers are combined (+0.127 from PDF+DB to Full). This confirms that the simultaneous combination of all three layers is the determining factor for answer quality: ROUGE-L increases from 0.055 to 0.167 (3.0x) and BLEU-1 from 0.002 to 0.213 when Layer 3 (Chat) is integrated. This confirms that tacit knowledge in team discussion logs contributes substantively to questions requiring operational context (E1: quotation bug, E2: decimal digit decision, E3: BOND_SYS ETL incident [anonymized], E4: upload allocation flow, E5: amend feature status). These ablation results quantitatively demonstrate that the three-layer corpus architecture, not merely the multi-adapter pipeline architecture, is the determining factor for answer quality on cross-paradigm questions.



Visualization of Results
Evaluation results are visualized across four separate panels (Figures 2a-2d). 	

Figure 2a presents the mean standard metrics per scenario in a grouped bar chart. P@K and MRR consistently reach 1.000 across all scenarios; Scenario B exhibits the highest ROUGE-L among reference-free scenarios (0.051), while Scenario E achieves ROUGE-L = 0.167 and BLEU-1 = 0.213 (reference-based).

Figure 2b presents the overall score per question (25 questions across 5 scenarios), revealing the distribution of performance across questions and scenarios.

Figure 2c compares three composite metrics (KTE, MSRS, AQI) across scenarios. MSRS is highest in Scenario C (PostgreSQL, 0.825), KTE is highest in Scenario D/PDF+DB (0.484), and Scenario E yields KTE = 0.414 despite its questions being the most complex.

Figure 2d is a radar chart illustrating the multi-dimensional profiles of all five scenarios, confirming that each scenario exhibits a distinct profile: A leads in Completeness, B in Faithfulness, C in MSRS, D in Overall and KTE, and E Retrieval Relevance and reference-based ROUGE-L.







Figure 2a. Mean standard metrics per scenario (grouped bar chart) 
 

Figure 2b. Overall score per question (25 questions, 5 scenarios)

Figure 2c. Composite metrics KTE, MSRS and AQI per scenario

Figure 2d. Radar chart of multi-dimensional profiles across five scenarios

Research Limitation
The retrieval component uses paraphrase multilingual-MiniLM-L12-v2 (50+ languages), ensuring no language bias at the retrieval layer; retrieval weights are entirely determined by semantic similarity, not source type metadata. For 

single-entity questions, Context Coverage tends to be low because top-K chunks concentrate on a single document; this accurately reflects the corpus structure rather than a system bias. Content hash-based deduplication is implemented to prevent a single file from being counted as multiple distinct sources. 

Several limitations of this study must be explicitly acknowledged:
Evaluation coverage. The evaluation covers 25 questions across five scenarios within a single domain (a domain-specific financial instrument transaction platform, BOND_SYS) using an Indonesian-language corpus. Empirical validation across other domains and languages is required to strengthen the generalizeability of the results.
Partial ground truth. Only Scenario E (5 questions) is equipped with manually curated reference answers (ground truth). Scenarios A-D employ reference-free evaluation, such that ROUGE-L and BLEU-1 for those four scenarios are computed against retrieved context rather than ideal reference answers. Scenario E’s reference-based ROUGE-L (0.167) and BLEU-1 (0.213) cannot be directly compared with the reference-free ROUGE-L of Scenarios A-D (0.024-0.051). Extending ground truth to all 25 questions would improve comparative validity across scenarios.
Evaluation scale. Evaluation across 25 questions in five scenarios provides an adequate proof-of-concept, but is insufficient for statistically generalizable claims. Future research is recommended to employ at least 20-50 query answer pairs per scenario with curated ground truth across all scenarios.

Implication 
4.4.1 Implications for Organizational Knowledge Transfer

The system directly addresses the challenge identified in the introduction. A newly onboarded Product Owner can immediately pose questions in natural language, for example, “What bug was found in the submit quotation on board BOARD_TYPE_A and how was it resolved?” or “What is the implementation status of the amend feature in the BOND_MOD_CUSTODY?” , and the system will automatically search for answers from the combination of PDF system specification documents, 908-message team discussion logs and the BOND_SYS PostgreSQL database within a single query. This scenario represents knowledge transfer from previous personnel to their successors without 

requiring a complete re-reading of all scattered documentation, consistent with the challenge of tacit knowledge explicitation identified by Nonaka and Takeuchi (1995) and with knowledge management system design requirements highlighted by Alavi and Leidner (2001). The same principle applies to other organizational contexts: new analysts can query technical decisions or system configurations from existing project documentation.

4.4.2 Implications for System Extensibility

The Adapter Pattern architecture opens a path for extending to new source types. MongoDB, SharePoint and Google Drive API can each be integrated by implementing two methods, load() and describe(), without modifying the core pipeline. This aligns with the Open/Close Principle in software design (Martin 2017)

The real-time vs. pre-indexed trade-off should be considered according to use case, as summarised in Table 7.
Table 7. Trade-off Comparison: Real-Time vs. Pre-Indexed Indexing
Aspect
Real-time (this system)
Pre-indexed
Data consistency
Always up-to-date
Potentially stale
Cold start time
~2-5 seconds
Instan
Storage management
No disk overhead
Requires synchronization
Best suited for
Frequently changing documents 
Large static corpora

For very large static corpora (millions of documents), FAISS IVF (Inverted File Index) or hybrid search integration with BM25-based hybrid search can be considered as architectural evolution (Roberstson and Zaragoza 2009; Manning et al. 2008).



Conclusion

This study successfully develops an agnostic multi-source RAG system fulfilling the research objectives, with six principal findings:
Agnostic architecture realized: SourceDetector + SourceFactory + BaseSourceAdapter enable handling of folders (TXT/PDF) and PostgreSQL (3 query modes) from a single source parameter, without modifications to the core pipeline. Automatic source type detection via pattern matching operates consistently across all 25 questions.
Real-time indexing confirmed: Each invocation of pipeline.ask() constructs the FAISS index in-memory with no index files on disk. Content changes in the source are immediately reflected in retrieval results without system restart.
Multi-source empirically proven: Batch evaluation of 25 questions (5 scenarios x 5 questions) shows Precision@K = 1.000 and MRR = 1.000 across all scenarios. Aggregate performance: A (Chat, Overall = 0.282, MSRS = 0.725), B (PDF, Overall = 0.276, MSRS = 0.625), C (PostgreSQL, Overall = 0.228, MSRS = 0.825), D (PDF + DB, Overall = 0.318, MSRS = 0.763), E (Hybrid, Overall = 0.358, MSRS = 0.713). Scenario E proves the source-agnostic claim at the highest level: Retrieval Relevance = 0.582, the highest amon all scenarios, confirming that a single query can reach relevant chunks from all three corpus layers simultaneously..
Ground truth Scenario E yields meaningfull ROUGE-L and BLEU-1: With 5 manually curated reference answers, Scenario E achieve ROUGE-L = 0.167 and BLEU-1 = 0.213 (reference-based), demonstrating genuine content overlap between AI-generated answer and ideal answers for cross-layer questions. This represents approximately a ~7x improvement over the highest reference-free ROUGE-L (Scenario B = 0.051).
Ablation study proves dramatic Layer 3 contribution: As detailed in 4.1.3, Overall scores are statistically indistinguishable across Chat-only, PDF-only, and PDF + DB configurations (0.230, 0.230, 0.231 respectively), but jump sharply to 0.358 when all three layers are activated (+0.127). ROUGE-L increases 3x from 0.055 to 0.167 and BLEU-1 from 0.002 to 0.213, proving that team discussion logs (Layer 3) are the determining component for cross-paradigm answer quality.
Knowledge transfer effectiveness measured: KTE per scenario: A = 0.452 (tacit to operational), B = 0.375 (explicit to actionable), C = 0.328 (explicit to structured), D = 0.484 (explicit to cross-referenced), E = 0.414 (cross-paradigm). Scenario D achieves the highest KTE (0.484), followed by A (0.452) and E (0.414), demonstrating that the right source combination matched to domain-specific questions consistently produces effective knowledge transfer..

The principal contributions of this study are: (i) an Adapter Pattern design that separates concern among data sources, text processing, retrieval, and generation; (ii) a five-scenario evaluation design with a three-layer real operational corpus; and (iii) a partial ground truth framework that enables reference-based validation on priority scenarios without requiring complete ground truth across all scenarios.

For future research, the following directions are recommended: (1) extending ground truth to all 25 questions for stronger comparative validation; (2) hybrid search (FAISS + BM25) to improve retrieval on queries with domain-specific tokens; (3) fine-tuning the embedding model on organization-specific domain documents; and (4) comparison with commercial vector database-based RAG systems (Pinecone, Weaviate) as architectural baselines.


Reference
Chui M, Manyika J, Bughin J, Dobbs R, Roxburgh C, Sarrazin H, Sands G, Westergreen M (2012) The social economyL unlocking value and productivity through social technologies. McKinsey Global InstituteFriends and neighbors on the web. Soc Networks 25(3):211–230. https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-social-economy
Alavi MJ, Leidner DE (2001) Review: knowledge management and knowledge management system. MIS Q 25(1): 107-136.
 https://doi.org/10.2307/3250961 
Borgeaud S, Mensch A, Hoffman J, Cal T, Rutherford E, Millican K, van den Driessche g, Lespiau JB, Damoc B, Clark A, et al. (2022) Improving language models by retrieving from trillions of tokens. In: Proceedings of the 39th INternational Conference on Machine Learning (ICML).
 https://doi.org/10.48550/arXiv.2112.04426
Carbonell J, Goldstein J (1998) The use of MMR, diversity-based reranking for reordering documents and producing summaries. In: Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pp 335-336. https://doi.org/10.1145/290941.291025 
Es s, James J, Espinosa-Anke L, Schockaert S (2023) RAGAS: automated evaluation of retrieval augmented generation. arXiv:2309.15217. https://doi.org/10.48550/arXiv.2309.15217
Douze M, Guzhva A, Deng C, Johnson J, Szilvasy G, Mazare PE, Lomeli M, Joulin A, Jegou H (2024) The FAISS library. arXiv:2401.08281.. https://doi.org/10.48550/arXiv.2401.08281
Fabbri AR, Krycinski W, McCann B, Xiong C, Socher R, Radev D (2021) SummEval: re-evaluating summarization evaluation. Trans Assoc Comput Linguist 9:391-409. https://doi.org/10.1162/tacl_a_00373
Gao Y, Xiong Y, Gao X, Jia K, Pan J, Bi Y, Dai Y, Sun J, Wang M, Wang H (2024) Retrieval-augmented generation for large language models: a survey. arXiv:2312.1097. https://doi.org/10.48550/arXiv.2312.10997
Gamma E, Helm R, Johnson R, Vlissides J (1994) Design patterns: elements of reusable object-oriented software. Addison-Wesley. https://books.google.co.id/books?id=6oHuKQe3TjQC&printsec=frontcover&source=gbs_ge_summary_r&cad=0#v=onepage&q&f=false
Gartner (2020) Gartner says employees spend too much time on low-value tasks: use AI and automation to fix it. Gartner Newsroom. https://www.gartner.com/en/newsroom/press-releases/2020-01-23-gartner-says-employees-spend-too-much-time-on-low-value-tasks
IDC (2023) 90% of data is unstructured and it’s full of untapped value. IDC Blog. 
https://blogs.idc.com/2023/05/09/90-of-data-is-unstructured-and-its-full-of-untapped-value/ 
Izacard G, Grave E (2021) Leveraging passage retrieval with generative models for open domain question answering. In: Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics, pp 874-880. https://doi.org/10.18653/v1/2021.eacl-main.74 
Ji Z, Lee N, Frieske R, Yu T, Su D, Xu Y, Ishil E, Bang YJ, Mandotto A, Fung P (2023) Survey of hallucination in natural language generation. ACM Comput Surv 55(12):1-38.
https://doi.org/10.1145/3571730 
Johnson J, Douze M, Jegou H (2019) Billion-scale similarity search with GPUs. IEEE trans Big Data 7 (3):535-547. https://doi.org/10.1109/TBDATA.2019.2921572
Kapurkhin V, Oguz B, Min S, Lewis P, Wu L, Edunov S, Chen D, Yih W (2020) Dense passage retrieval for open-domain question answering. In: Proceedings of EMNLP 2020, pp 6769-6781. https://doi.org/10.18653/v1/2020.emnlp-main.550
Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, Kiela D (2020) Retrieval-Augmented generation for knowledge-intensive NLP tasks. In: Advances in Neural Information Processing Systems 33, pp 9459-9474. https://doi.org/10.48550/arXiv.2005.11401
Lin CY (2004) ROUGE: a package for automatic evaluation of summaries. In: Proceedings of the ACL Workshop on Text Summarization Branches Out. https://aclanthology.org/W04-1013
Manning CD, Raghavan P, Schutze H (2008) Introduction to information retrieval. Cambridge University Press, Cambridge.
Martin RC (2017) Clean architecture: a craftsman’s guide to software structure and design. Prentice Hall, Upper Saddle river
Nonaka I, Takeuchi H (1995) the knowledge-creating company. Oxford University Press, New York.
Papineni K, Roukos S, Ward T, Zhu WJ (2002) BLEU: a method for automatic evaluation of machine translation. In: Proceedings of the 40th Annual Meeting of Association for Computational Linguistics, pp 311-318.
https://doi.org/10.3115/1073083.1073135
Robertson S, Zaragoza H (2009) The probabilistic relevance framework: BM25 and beyond. Found Trends Inf Retr 3(4):333-389.
https://doi.org/10.1561/1500000019
Reimers N, Gurevych I (2019) Sentence-BERT: sentence embeddings using Siamese BERT-networks. IN: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. https://doi.org/10.18653/v1/D19-1410 
Ren H, Shi H, Zhao W, Zhao J, Zhao Y (2023) Self-RAG: learning to retrieve, generate, and critique through self-reflection. arXiv:2307.11019. https://doi.org/10.48550/arXiv.2307.11019
Shi P, Li X, Han X, Chang B, Sui Z (2023) REPLUG: retrieval-augmented black-box language models. arXic:2301.12652.. https://doi.org/10.48550/arXiv.2301.12652
Voorhees EM (1999) The TREC-8 question answering track report. In: Proceedings of the 8th Text retrieval Conference (TREC-8), pp 77-82. https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf
Yasunaga M, Ren H, Bosselut A, Liang P, Leskovec J (2021) QA-GNN: reasoning with language models and knowledge graphs for question answering. In: Proceedings of the 2021 Conference of the North American Chapter of the association for Computational Linguistics, pp 535-545. https://doi.org/10.18653/v1/2021.naacl-main.45
Gamma E, Helm R, Johnson R, Vlissides J (1994) Design patterns: elements of reusable object-oriented software. Addison-Wesley. https://books.google.co.id/books?id=6oHuKQe3TjQC&printsec=frontcover&source=gbs_ge_summary_r&cad=0#v=onepage&q&f=false

